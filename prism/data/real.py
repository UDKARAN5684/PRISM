"""Real-world public-dataset adapters for the PRISM pipeline.

PRISM is built around a ground-truth simulator, which is the only way to score a causal
estimator with PEHE. The obvious criticism of that design is "it is just a simulation".
This module answers it: the *same* downstream pipeline (feature store, survival models,
uplift learners, budget optimiser) can be pointed at genuinely public, no-authentication
datasets by translating them into the canonical gold-panel column names defined in
:mod:`prism.data.schema`.

Three sources are supported, chosen so that between them they exercise every leg of the
project:

============================  ===========================================================
``telco_churn``               labelled churn + tenure -> survival / churn_next
``online_retail_ii``          raw transactions -> RFM, CLV, point-in-time features
``hillstrom``                 a real *randomised* three-arm experiment -> uplift evaluation
============================  ===========================================================

Design rules honoured here
--------------------------
* **Never raises on the network.** :func:`load_real` returns ``None`` and logs a warning
  for any failure whatsoever, so an offline run of ``prism.pipelines.run_all`` still works.
* **Every HTTP call has a timeout** (default 20 s, hard-capped at
  :data:`MAX_TIMEOUT_SECONDS`) and a whole-body deadline.
* **Cache first.** Raw bytes *and* a parsed parquet are written under ``cache_dir``, with a
  JSON sidecar recording url, sha256, byte count and shape. Later calls read the cache and
  touch the network only if the cache is missing or ``force_refresh=True``.
* **No fabricated data.** :func:`map_to_panel` fills canonical columns the source genuinely
  cannot supply with ``NaN`` and lists them in ``df.attrs["unmapped_columns"]``. Anything
  computed rather than read verbatim is recorded in ``df.attrs["derivations"]`` so a
  reviewer can audit exactly what was assumed.

Examples
--------
>>> from prism.data.real import describe_datasets, load_real, map_to_panel
>>> describe_datasets()[["name", "n_rows_approx"]]        # doctest: +SKIP
>>> raw = load_real("hillstrom")                          # doctest: +SKIP
>>> panel = map_to_panel(raw, "hillstrom")                # doctest: +SKIP
>>> panel.attrs["unmapped_columns"]                       # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PANEL_COLUMNS,
    PANEL_DTYPES,
)
from prism.utils.logging import get_logger
from prism.utils.seeds import as_rng

__all__ = [
    "AVAILABLE_DATASETS",
    "HILLSTROM_ARM_MAP",
    "DEFAULT_CACHE_DIR",
    "MAX_TIMEOUT_SECONDS",
    "RETAIL_LAPSE_DAYS",
    "load_real",
    "map_to_panel",
    "load_real_panel",
    "describe_datasets",
    "dataset_status",
    "clear_cache",
]

log = get_logger("data.real")

#: Where parsed frames and raw payloads are cached (relative paths resolve to the CWD).
DEFAULT_CACHE_DIR: str = "artifacts/data/real"

#: Hard ceiling on any socket timeout. The pipeline must never block a run on the network.
MAX_TIMEOUT_SECONDS: float = 25.0

_DAYS_PER_MONTH: float = 30.4375
_USER_AGENT: str = "prism-research/1.0 (+https://example.invalid/prism) python-urllib"

#: Days of inactivity after which an Online Retail II customer is *labelled* lapsed.
#: This is a convention, not a fact in the source; it is reported in ``derivations``.
RETAIL_LAPSE_DAYS: int = 90


# ======================================================================================
# Dataset registry
# ======================================================================================
AVAILABLE_DATASETS: dict[str, dict[str, Any]] = {
    "telco_churn": {
        "name": "telco_churn",
        "title": "IBM Telco Customer Churn",
        "urls": (
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv",
        ),
        "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "format": "csv",
        "description": (
            "7,043 telecom subscribers with a binary churn label, tenure in months, contract "
            "type and monthly/total charges. The canonical teaching dataset for churn."
        ),
        "demonstrates": (
            "supervised churn labels and right-censored tenure -> prism.models.survival "
            "(discrete-time hazard, Cox, concordance) on non-simulated data"
        ),
        "citation": "IBM Cognos Analytics sample, 'Telco customer churn' (IBM Community, 2018).",
        "license": "Apache-2.0 (as redistributed by the IBM/telco-customer-churn-on-icp4d repo)",
        "n_rows_approx": 7_043,
        "n_cols_approx": 21,
        "size_mb_approx": 0.95,
        "entity": "customer (one row per customer, cross-section)",
        "randomised": False,
        "download_hint": "about 1 MB; a 20 s budget is ample.",
        "suggested_timeout": 20.0,
        "requires": (),
    },
    "online_retail_ii": {
        "name": "online_retail_ii",
        "title": "UCI Online Retail II",
        "urls": ("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",),
        "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "format": "zip:xlsx",
        "description": (
            "About 1.07M invoice lines from a UK online giftware retailer, 2009-12 to 2011-12. "
            "Transactional, so it is the honest test of the point-in-time feature store."
        ),
        "demonstrates": (
            "raw event log -> RFM aggregation, BG/NBD + Gamma-Gamma CLV "
            "(prism.models.clv) and as-of feature joins (prism.data.features)"
        ),
        "citation": (
            "Chen, D. (2019). Online Retail II. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5CG6D"
        ),
        "license": "CC BY 4.0",
        "n_rows_approx": 1_067_371,
        "n_cols_approx": 8,
        "size_mb_approx": 44.0,
        "entity": "invoice line (aggregated to one row per customer by map_to_panel)",
        "randomised": False,
        "download_hint": "about 45 MB plus a 2-3 minute openpyxl parse; pass max_seconds=180 and expect the first call to be slow. Later calls read the parquet cache in under a second.",
        "suggested_timeout": 25.0,
        "requires": ("openpyxl",),
    },
    "hillstrom": {
        "name": "hillstrom",
        "title": "Hillstrom MineThatData E-Mail Analytics Challenge",
        "urls": (
            "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
            "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
            "https://raw.githubusercontent.com/uber/causalml/master/causalml/dataset/data/Hillstrom.csv",
        ),
        "url": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        "format": "csv",
        "description": (
            "64,000 customers randomly assigned to one of three arms (No E-Mail / Mens E-Mail / "
            "Womens E-Mail) with visit, conversion and spend observed over the next two weeks."
        ),
        "demonstrates": (
            "a genuine multi-arm RCT: Qini/AUUC, GATES, BLP calibration and doubly-robust "
            "policy value (prism.causal.evaluate) computed on real experimental data, where "
            "randomisation makes the propensity known rather than estimated"
        ),
        "citation": (
            "Hillstrom, K. (2008). The MineThatData E-Mail Analytics And Data Mining Challenge. "
            "blog.minethatdata.com"
        ),
        "license": "Public, free to use with attribution (per the author's challenge posting)",
        "n_rows_approx": 64_000,
        "n_cols_approx": 12,
        "size_mb_approx": 3.8,
        "entity": "customer (one row per randomised customer)",
        "randomised": True,
        "download_hint": "about 4 MB from minethatdata.com over plain HTTP; a 20 s budget is ample.",
        "suggested_timeout": 20.0,
        "requires": (),
    },
}

#: Hillstrom's ``segment`` string -> PRISM arm index. Control is 0, as everywhere in PRISM.
HILLSTROM_ARM_MAP: dict[str, int] = {
    "no e-mail": 0,
    "womens e-mail": 1,
    "mens e-mail": 2,
}


# ======================================================================================
# Small internal helpers
# ======================================================================================
def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string (metadata only, never logic)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _cache_paths(name: str, cache_dir: str | Path) -> dict[str, Path]:
    """Return the cache file locations for one dataset.

    Parameters
    ----------
    name : str
        Dataset key from :data:`AVAILABLE_DATASETS`.
    cache_dir : str or pathlib.Path
        Directory holding the cache.

    Returns
    -------
    dict of str to pathlib.Path
        Keys ``dir``, ``raw``, ``parquet`` and ``meta``.
    """
    root = Path(cache_dir)
    spec = AVAILABLE_DATASETS.get(name, {})
    ext = ".zip" if str(spec.get("format", "csv")).startswith("zip") else ".csv"
    return {
        "dir": root,
        "raw": root / f"{name}.raw{ext}",
        "parquet": root / f"{name}.parquet",
        "meta": root / f"{name}.meta.json",
    }


def _read_meta(path: Path) -> dict[str, Any]:
    """Read a cache sidecar, returning ``{}`` if it is missing or corrupt."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_meta(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a cache sidecar, swallowing filesystem errors (the cache is an optimisation)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(dict(payload), fh, indent=2, sort_keys=True)
    except Exception as exc:  # pragma: no cover - depends on filesystem permissions
        log.warning("could not write cache metadata %s: %s", path, exc)


def _http_get(url: str, timeout: float, max_seconds: float | None = None) -> bytes | None:
    """Fetch ``url`` and return its body, or ``None`` on any failure.

    Uses :mod:`requests` when installed and falls back to :mod:`urllib.request`. The socket
    timeout is clamped to :data:`MAX_TIMEOUT_SECONDS`; ``max_seconds`` additionally bounds
    the total time spent streaming the body, so a stalled-but-alive connection cannot hang
    the pipeline.

    Parameters
    ----------
    url : str
        Absolute HTTP(S) URL.
    timeout : float
        Per-socket timeout in seconds, clamped to ``(0, MAX_TIMEOUT_SECONDS]``.
    max_seconds : float, optional
        Whole-transfer deadline. Defaults to the clamped ``timeout``.

    Returns
    -------
    bytes or None
        The response body, or ``None`` if anything at all went wrong.
    """
    t_sock = float(min(max(float(timeout), 0.1), MAX_TIMEOUT_SECONDS))
    budget = float(max_seconds if max_seconds is not None else t_sock)
    started = time.monotonic()
    deadline = started + budget
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}

    try:
        try:
            import requests  # noqa: PLC0415 - optional accelerator, imported at point of use

            with requests.get(url, timeout=t_sock, headers=headers, stream=True) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                for chunk in resp.iter_content(chunk_size=1 << 18):
                    if chunk:
                        chunks.append(chunk)
                    if time.monotonic() > deadline:
                        raise TimeoutError(  # noqa: B904 - raised from the try body, not an except clause
                            f"exceeded the {budget:.0f}s transfer budget"
                        )
                body = b"".join(chunks)
        except ImportError:  # pragma: no cover - requests is normally installed
            import urllib.request  # noqa: PLC0415

            req = urllib.request.Request(url, headers=headers)
            chunks = []
            with urllib.request.urlopen(req, timeout=t_sock) as resp:  # noqa: S310 - fixed https/http registry
                while True:
                    chunk = resp.read(1 << 18)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if time.monotonic() > deadline:
                        # noqa below: this raise is in the try body, not an except
                        # clause -- there is no active exception to chain from.
                        raise TimeoutError(  # noqa: B904
                            f"exceeded the {budget:.0f}s transfer budget"
                        )
            body = b"".join(chunks)
    except Exception as exc:
        log.warning("download failed (%s) for %s: %s", type(exc).__name__, url, exc)
        return None

    if not body:
        log.warning("download returned an empty body for %s", url)
        return None
    log.info("downloaded %.2f MB in %.1fs from %s", len(body) / 1e6, time.monotonic() - started, url)
    return body


def _parse_bytes(name: str, raw: bytes) -> pd.DataFrame | None:
    """Parse cached raw bytes for ``name`` into a DataFrame, or ``None`` on failure.

    Parameters
    ----------
    name : str
        Dataset key.
    raw : bytes
        Payload as downloaded (CSV text, or a ZIP containing one workbook).

    Returns
    -------
    pandas.DataFrame or None
    """
    fmt = str(AVAILABLE_DATASETS[name]["format"])
    try:
        if fmt == "csv":
            df = pd.read_csv(io.BytesIO(raw), encoding_errors="replace", low_memory=False)
        elif fmt == "zip:xlsx":
            df = _parse_retail_zip(raw)
        else:  # pragma: no cover - registry is closed
            raise ValueError(f"unhandled format {fmt!r} for dataset {name!r}")
    except ImportError as exc:
        log.warning(
            "cannot parse %s: missing optional dependency (%s). Install it with: pip install openpyxl",
            name,
            exc,
        )
        return None
    except Exception as exc:
        log.warning("failed to parse %s (%s): %s", name, type(exc).__name__, exc)
        return None

    if df is None or df.empty:
        log.warning("parsed frame for %s is empty", name)
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_retail_zip(raw: bytes) -> pd.DataFrame:
    """Extract and concatenate every sheet of the Online Retail II workbook inside a ZIP.

    Parameters
    ----------
    raw : bytes
        The ``online+retail+ii.zip`` payload.

    Returns
    -------
    pandas.DataFrame
        All invoice lines from both year sheets, with a ``source_sheet`` column.

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed (the caller converts this into ``None``).
    ValueError
        If the archive contains no workbook.
    """
    import importlib

    try:
        importlib.import_module("openpyxl")
    except ImportError as exc:  # re-raised as ImportError for the caller's handler
        raise ImportError("openpyxl") from exc

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".xlsx", ".xls"))]
        if not members:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not csv_members:
                raise ValueError(f"no workbook or csv inside archive; members={zf.namelist()[:5]}")
            with zf.open(csv_members[0]) as fh:
                return pd.read_csv(fh, low_memory=False)
        payload = zf.read(members[0])

    log.info("parsing the Online Retail II workbook (about 1.07M rows); this takes a couple of minutes")
    sheets = pd.read_excel(io.BytesIO(payload), sheet_name=None, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    for sheet_name, frame in sheets.items():
        frame = frame.copy()
        frame["source_sheet"] = str(sheet_name)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    # Invoice and StockCode mix ints and strings ("C489449" marks a credit note), which
    # pyarrow cannot infer; pin them to string so the parquet cache can be written.
    for col in ("Invoice", "StockCode", "Description"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def _write_parquet_cache(df: pd.DataFrame, path: Path, name: str) -> bool:
    """Write the parsed frame to parquet, retrying once with object columns stringified.

    Real CSV/XLSX sources routinely mix ints and strings inside one column (Online Retail
    II's ``Invoice`` carries ``"C489449"`` credit notes), which Arrow refuses to infer. A
    failed cache write is not fatal, but it would silently force a multi-minute re-parse on
    every call, so the retry matters.

    Parameters
    ----------
    df : pandas.DataFrame
        Parsed source frame.
    path : pathlib.Path
        Destination parquet file.
    name : str
        Dataset key, for logging.

    Returns
    -------
    bool
        True if the cache was written.
    """
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        log.info("%s: parquet write needs a dtype repair (%s)", name, str(exc)[:120])
    try:
        safe = df.copy()
        for col in safe.columns:
            if safe[col].dtype == object:
                safe[col] = safe[col].astype("string")
        safe.to_parquet(path, index=False)
        log.info("%s: parquet cache written after stringifying mixed-type columns", name)
        return True
    except Exception as exc:
        log.warning("%s: could not write parquet cache (%s); the raw cache is still used", name, str(exc)[:160])
        return False


def _subsample(
    df: pd.DataFrame, sample_rows: int, random_state: int | np.random.Generator | None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Deterministically take at most ``sample_rows`` rows, preserving source order.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to thin out.
    sample_rows : int
        Maximum number of rows to keep.
    random_state : int or numpy.random.Generator or None
        Seed material, normalised through :func:`prism.utils.seeds.as_rng`.

    Returns
    -------
    tuple of (pandas.DataFrame, numpy.ndarray)
        The thinned frame and the **original row positions** it was taken from. Adapters
        that have to synthesise a customer key (Hillstrom ships none) use those positions
        so a subsampled load and a full load agree on who is who.
    """
    if sample_rows <= 0 or len(df) <= sample_rows:
        return df, np.arange(len(df), dtype="int64")
    rng = as_rng(random_state)
    idx = np.sort(rng.choice(len(df), size=int(sample_rows), replace=False)).astype("int64")
    return df.iloc[idx].reset_index(drop=True), idx


# ======================================================================================
# Public API: loading
# ======================================================================================
def load_real(
    name: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    timeout: float = 20.0,
    max_seconds: float | None = None,
    force_refresh: bool = False,
    allow_download: bool = True,
    sample_rows: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame | None:
    """Load one public dataset, from cache when possible, otherwise over HTTP.

    This function **never raises**. Every failure mode -- unknown name, no network, DNS
    failure, HTTP error, truncated body, unparsable payload, missing ``openpyxl``,
    unwritable cache -- is logged and turns into ``None`` so that an offline run of the
    PRISM pipeline continues on simulated data.

    Parameters
    ----------
    name : str
        One of ``"telco_churn"``, ``"online_retail_ii"``, ``"hillstrom"``.
    cache_dir : str or pathlib.Path, default ``"artifacts/data/real"``
        Directory for the raw payload, the parsed parquet and the JSON sidecar. Created if
        absent. Relative paths resolve against the current working directory.
    timeout : float, default 20.0
        Per-socket timeout in seconds, clamped to :data:`MAX_TIMEOUT_SECONDS`.
    max_seconds : float, optional
        Whole-transfer deadline in seconds. Defaults to the clamped ``timeout``. Raise it
        for ``online_retail_ii`` (about 44 MB) on a slow link.
    force_refresh : bool, default False
        Ignore any cached copy and re-download.
    allow_download : bool, default True
        If False, only the cache is consulted -- useful for deliberately offline tests.
    sample_rows : int, optional
        Deterministically thin the returned frame to at most this many rows. The full frame
        is still cached.
    random_state : int or numpy.random.Generator or None, optional
        Seed for ``sample_rows``; normalised via :func:`prism.utils.seeds.as_rng`.

    Returns
    -------
    pandas.DataFrame or None
        The parsed source frame with ``df.attrs`` populated (``dataset``, ``source_url``,
        ``citation``, ``license``, ``cached``, ``cache_dir``), or ``None`` on any failure.

    Examples
    --------
    >>> df = load_real("telco_churn", timeout=5)      # doctest: +SKIP
    >>> None if df is None else df.shape              # doctest: +SKIP
    (7043, 21)
    """
    spec = AVAILABLE_DATASETS.get(name)
    if spec is None:
        log.error("unknown dataset %r; available: %s", name, sorted(AVAILABLE_DATASETS))
        return None

    paths = _cache_paths(name, cache_dir)
    try:
        paths["dir"].mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("cannot create cache dir %s: %s (continuing without cache)", paths["dir"], exc)

    df: pd.DataFrame | None = None
    from_cache = False

    # ---- 1. parsed parquet cache ------------------------------------------------------
    if not force_refresh and paths["parquet"].exists():
        try:
            df = pd.read_parquet(paths["parquet"])
            from_cache = True
            log.info("%s: loaded %d rows from parquet cache", name, len(df))
        except Exception as exc:
            log.warning("%s: parquet cache unreadable (%s); will re-parse", name, exc)
            df = None

    # ---- 2. raw bytes cache -----------------------------------------------------------
    raw: bytes | None = None
    if df is None and not force_refresh and paths["raw"].exists():
        try:
            raw = paths["raw"].read_bytes()
            from_cache = True  # served without touching the network, same as the parquet path
            log.info("%s: re-parsing %.2f MB from raw cache", name, len(raw) / 1e6)
        except Exception as exc:
            log.warning("%s: raw cache unreadable (%s)", name, exc)
            raw = None

    # ---- 3. network -------------------------------------------------------------------
    if df is None and raw is None:
        if not allow_download:
            log.warning("%s: not cached and allow_download=False -> returning None", name)
            return None
        for url in spec["urls"]:
            raw = _http_get(url, timeout=timeout, max_seconds=max_seconds)
            if raw is not None:
                spec_used_url = url
                break
        else:
            log.warning(
                "%s: every candidate URL failed (%d tried). Running offline; returning None.",
                name,
                len(spec["urls"]),
            )
            return None
        try:
            paths["raw"].write_bytes(raw)
        except Exception as exc:
            log.warning("%s: could not write raw cache (%s)", name, exc)
        _write_meta(
            paths["meta"],
            {
                "dataset": name,
                "source_url": spec_used_url,
                "fetched_at": _now_iso(),
                "raw_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "citation": spec["citation"],
                "license": spec["license"],
            },
        )

    # ---- 4. parse + cache parquet -----------------------------------------------------
    if df is None:
        if raw is None:  # pragma: no cover - unreachable, kept as a guard
            return None
        df = _parse_bytes(name, raw)
        if df is None:
            return None
        _write_parquet_cache(df, paths["parquet"], name)
        meta = _read_meta(paths["meta"])
        meta.update({"dataset": name, "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
                     "parsed_at": _now_iso(), "columns": [str(c) for c in df.columns]})
        _write_meta(paths["meta"], meta)

    positions = np.arange(len(df), dtype="int64")
    if sample_rows is not None:
        df, positions = _subsample(df, int(sample_rows), random_state)

    df = df.reset_index(drop=True)
    df.attrs.update(
        {
            "dataset": name,
            "title": spec["title"],
            "source_url": _read_meta(paths["meta"]).get("source_url", spec["url"]),
            "citation": spec["citation"],
            "license": spec["license"],
            "cached": bool(from_cache),
            "cache_dir": str(paths["dir"]),
            "n_source_rows": int(len(df)),
            # Original row positions in the full source, so a subsampled load and a full
            # load synthesise the same customer key for the same underlying customer.
            "source_positions": positions,
        }
    )
    return df


# ======================================================================================
# Public API: mapping onto the canonical panel
# ======================================================================================
def _blank_panel(n: int, index: pd.Index | None = None) -> pd.DataFrame:
    """Create an all-missing frame with exactly the canonical panel columns.

    Parameters
    ----------
    n : int
        Number of rows.
    index : pandas.Index, optional
        Index to attach; defaults to a fresh ``RangeIndex``.

    Returns
    -------
    pandas.DataFrame
        ``n`` rows, columns :data:`prism.data.schema.PANEL_COLUMNS`, every value missing.
    """
    idx = pd.RangeIndex(n) if index is None else index
    data: dict[str, pd.Series] = {}
    for col in PANEL_COLUMNS:
        dtype = PANEL_DTYPES.get(col, "float64")
        if dtype == "datetime64[ns]":
            data[col] = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
        elif dtype in ("object", "string"):
            data[col] = pd.Series(pd.NA, index=idx, dtype="object")
        else:
            data[col] = pd.Series(np.nan, index=idx, dtype="float64")
    return pd.DataFrame(data, index=idx)


def _coerce_panel_types(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical dtypes, demoting integer columns that still contain missing values.

    :func:`prism.data.schema.enforce_schema` casts panel integer columns straight to
    ``int64``, which is correct for the simulator but cannot represent the ``NaN`` that a
    real source legitimately leaves behind (Telco has no treatment arm, for instance). This
    helper therefore keeps such columns as ``float64`` and only narrows to ``int64`` when a
    column is complete.

    Parameters
    ----------
    panel : pandas.DataFrame
        Frame with canonical column names.

    Returns
    -------
    pandas.DataFrame
        Copy with canonical dtypes where possible.
    """
    out = panel.copy()
    for col, dtype in PANEL_DTYPES.items():
        if col not in out.columns:
            continue
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype in ("object", "string"):
            values = out[col].astype("object")
            if col in CATEGORICAL_FEATURES:
                values = values.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
            out[col] = values.where(values.notna(), None)
        elif dtype == "int64":
            num = pd.to_numeric(out[col], errors="coerce")
            out[col] = num.astype("int64") if num.notna().all() else num.astype("float64")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _finalise(
    panel: pd.DataFrame,
    extras: pd.DataFrame | None,
    *,
    name: str,
    mapped: Mapping[str, str],
    derivations: Mapping[str, str],
    notes: Sequence[str] = (),
    source_attrs: Mapping[str, Any] | None = None,
    leaky_features: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Assemble the returned frame and attach the provenance attributes.

    Parameters
    ----------
    panel : pandas.DataFrame
        Canonical columns, already populated where the source allowed.
    extras : pandas.DataFrame or None
        Source-specific columns to keep alongside (they are prefixed ``src_``).
    name : str
        Dataset key.
    mapped : mapping of str to str
        Canonical column -> plain-English provenance, for every column actually filled.
    derivations : mapping of str to str
        Canonical column -> the rule used to compute it, for anything not read verbatim.
    notes : sequence of str
        Free-text caveats surfaced in ``df.attrs["notes"]``.
    source_attrs : mapping, optional
        ``attrs`` carried over from the source frame.
    leaky_features : mapping of str to sequence of str, optional
        Outcome column -> canonical feature names that are mechanically tied to it in this
        source and MUST be dropped from ``X`` before modelling that outcome. Surfaced as
        ``df.attrs["leaky_features"]``. An empty mapping means "nothing in the feature
        block leaks", which is an assertion, not an omission.

    Returns
    -------
    pandas.DataFrame
        Canonical columns in :data:`prism.data.schema.PANEL_COLUMNS` order, then extras.
    """
    panel = panel.reindex(columns=list(PANEL_COLUMNS))
    panel = _coerce_panel_types(panel)

    if extras is not None and len(extras.columns):
        extras = extras.copy()
        extras.columns = [c if str(c).startswith("src_") else f"src_{c}" for c in extras.columns]
        extras.index = panel.index
        out = pd.concat([panel, extras], axis=1)
    else:
        out = panel

    unmapped = [c for c in PANEL_COLUMNS if c not in mapped]
    spec = AVAILABLE_DATASETS[name]
    src = dict(source_attrs or {})
    out.attrs.update(
        {
            "dataset": name,
            "title": spec["title"],
            "source_url": src.get("source_url", spec["url"]),
            "citation": spec["citation"],
            "license": spec["license"],
            "randomised": bool(spec["randomised"]),
            "unmapped_columns": unmapped,
            "mapped_columns": dict(mapped),
            "derivations": dict(derivations),
            "extra_columns": [c for c in out.columns if str(c).startswith("src_")],
            "n_source_rows": int(src.get("n_source_rows", len(out))),
            "notes": list(notes),
            "leaky_features": {k: list(v) for k, v in dict(leaky_features or {}).items()},
        }
    )
    log.info(
        "%s -> panel: %d rows, %d/%d canonical columns populated (%d unmapped)",
        name,
        len(out),
        len(mapped),
        len(PANEL_COLUMNS),
        len(unmapped),
    )
    return out


def _require_columns(df: pd.DataFrame, needed: Iterable[str], name: str) -> None:
    """Raise a clear :class:`ValueError` if ``df`` lacks any of ``needed``."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"frame does not look like the {name!r} source: missing columns {missing}; "
            f"got {list(df.columns)[:12]}"
        )


def _map_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Map IBM Telco Customer Churn onto the canonical panel.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw Telco frame as returned by :func:`load_real`.

    Returns
    -------
    pandas.DataFrame
        Canonical panel columns plus ``src_*`` passthroughs.
    """
    _require_columns(df, ["customerID", "tenure", "Contract", "Churn", "MonthlyCharges"], "telco_churn")
    n = len(df)
    panel = _blank_panel(n)

    tenure = pd.to_numeric(df["tenure"], errors="coerce").astype("float64")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").astype("float64")
    total = pd.to_numeric(df.get("TotalCharges"), errors="coerce").astype("float64")
    churn = pd.to_numeric(
        df["Churn"].astype("string").str.strip().str.lower().map({"yes": 1.0, "no": 0.0}),
        errors="coerce",
    ).astype("float64")

    panel["customer_id"] = df["customerID"].astype("string")
    panel["period"] = 0.0
    panel["tenure_months"] = tenure
    panel["avg_order_value"] = monthly
    panel["monetary_12m"] = monthly * tenure.clip(upper=12.0)
    panel["churn_next"] = churn
    panel["event_time"] = tenure
    panel["event_observed"] = churn

    contract = df["Contract"].astype("string").str.strip().str.lower()
    panel["contract_type"] = np.where(contract.eq("month-to-month"), "monthly", "annual")
    panel.loc[contract.isna(), "contract_type"] = None

    mapped: dict[str, str] = {
        "customer_id": "customerID",
        "period": "single cross-section, so every row is period 0",
        "tenure_months": "tenure (months with the company)",
        "avg_order_value": "MonthlyCharges (the monthly bill is the order value here)",
        "monetary_12m": "MonthlyCharges x min(tenure, 12)",
        "churn_next": "Churn (Yes/No) -> 1/0",
        "event_time": "tenure, months observed until churn or censoring",
        "event_observed": "Churn == Yes",
        "contract_type": "Contract: Month-to-month -> monthly, One/Two year -> annual",
    }
    derivations: dict[str, str] = {
        "period": "constant 0; the source is a cross-section with no time index",
        "monetary_12m": "MonthlyCharges x min(tenure, 12): trailing-12-month spend implied by the current bill",
        "contract_type": "'One year' and 'Two year' both collapse to the canonical level 'annual'",
        "event_observed": "Churn is 'churned last month', reused as the event indicator for tenure",
    }

    if "PaymentMethod" in df.columns:
        pay = df["PaymentMethod"].astype("string").str.lower()
        autopay = np.where(pay.str.contains("automatic", na=False), "yes", "no")
        panel["is_autopay"] = pd.Series(autopay, index=panel.index).where(pay.notna().to_numpy(), None)
        mapped["is_autopay"] = "PaymentMethod containing 'automatic'"
        derivations["is_autopay"] = "bank transfer/credit card (automatic) -> yes, mailed check/electronic check -> no"

    service_cols = [
        c
        for c in (
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        )
        if c in df.columns
    ]
    if service_cols:
        subscribed = pd.DataFrame(
            {c: df[c].astype("string").str.strip().str.lower().isin(["yes", "dsl", "fiber optic"]) for c in service_cols}
        )
        n_services = subscribed.sum(axis=1).astype("float64")
        panel["n_categories_12m"] = n_services
        panel["basket_diversity"] = n_services / float(len(service_cols))
        mapped["n_categories_12m"] = f"count of subscribed services across {len(service_cols)} service columns"
        mapped["basket_diversity"] = "subscribed services / available services"
        derivations["n_categories_12m"] = "a 'Yes' (or an internet type) in any service column counts as one category"
        derivations["basket_diversity"] = "share of the available service catalogue the customer takes"

    keep = [c for c in ("gender", "SeniorCitizen", "Partner", "Dependents", "PaymentMethod",
                        "PaperlessBilling", "InternetService", "Contract") if c in df.columns]
    extras = df[keep].copy()
    extras["TotalCharges_numeric"] = total

    notes = [
        "Telco is observational with no offer/treatment, so arm, treated and offer_cost are NaN: "
        "uplift learners cannot be scored here, only churn and survival models.",
        "No timestamps exist in the source, so as_of_date, cohort and the temporal split are NaN.",
        "TotalCharges contains blank strings for tenure=0 customers; kept as src_TotalCharges_numeric with NaN.",
        "LEAKAGE GUARD: event_time IS the tenure_months feature -- the source gives one tenure "
        "number and nothing else time-like. Fitting prism.models.survival with tenure_months in X "
        "therefore yields a concordance index of 1.0 that means nothing. Drop the columns listed in "
        "attrs['leaky_features']['event_time'] before any survival fit; churn_next is unaffected, "
        "tenure is a legitimate predictor of the binary label.",
        f"{int((tenure <= 0).sum())} customers have tenure 0 (all of them non-churners): a "
        "zero-length right-censored spell that contributes no information to a survival fit.",
    ]
    leaky = {
        "event_time": ["tenure_months", "monetary_12m"],
        "event_observed": ["tenure_months", "monetary_12m"],
    }
    return _finalise(panel, extras, name="telco_churn", mapped=mapped, derivations=derivations,
                     notes=notes, source_attrs=df.attrs, leaky_features=leaky)


def _map_hillstrom(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Hillstrom e-mail RCT onto the canonical panel.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw 64k-row Hillstrom frame as returned by :func:`load_real`.

    Returns
    -------
    pandas.DataFrame
        Canonical panel columns plus ``src_*`` passthroughs (including the experiment's
        own outcomes ``src_visit``, ``src_conversion``, ``src_spend``).
    """
    _require_columns(df, ["segment", "visit", "conversion", "spend", "recency", "history"], "hillstrom")
    n = len(df)
    panel = _blank_panel(n)

    segment = df["segment"].astype("string").str.strip().str.lower()
    arm = pd.to_numeric(segment.map(HILLSTROM_ARM_MAP), errors="coerce").astype("float64")
    unknown = int(arm.isna().sum())
    if unknown:
        log.warning("hillstrom: %d rows have an unrecognised segment label; arm left as NaN", unknown)

    spend = pd.to_numeric(df["spend"], errors="coerce").astype("float64")
    # `visit` and `conversion` are Hillstrom's other two outcomes. PRISM maps `spend`,
    # and the raw columns are preserved as src_visit / src_conversion by _finalise, so
    # a caller can target them instead without this adapter having to choose.
    recency_m = pd.to_numeric(df["recency"], errors="coerce").astype("float64")
    history = pd.to_numeric(df["history"], errors="coerce").astype("float64")

    # The source ships no identifier, so the key is the row's position in the *full* file.
    # load_real records the original positions in attrs, which keeps the key stable when
    # the frame has been thinned with sample_rows (otherwise two different customers would
    # both be called "hillstrom_000000").
    positions = np.asarray(df.attrs.get("source_positions", range(n)), dtype="int64")
    if positions.shape[0] != n:
        positions = np.arange(n, dtype="int64")
    panel["customer_id"] = pd.Series([f"hillstrom_{i:06d}" for i in positions], index=panel.index, dtype="string")
    panel["period"] = 0.0
    panel["assignment_block"] = "rct"
    panel["arm"] = arm
    panel["treated"] = (arm > 0).astype("float64").where(arm.notna())
    panel["recency_days"] = recency_m * _DAYS_PER_MONTH
    panel["monetary_12m"] = history
    panel["value_h"] = spend

    mapped: dict[str, str] = {
        "customer_id": "synthesised positional key (the source ships no identifier)",
        "period": "single two-week outcome window, so every row is period 0",
        "assignment_block": "constant 'rct': all 64,000 customers were randomised",
        "arm": "segment: No E-Mail -> 0, Womens E-Mail -> 1, Mens E-Mail -> 2",
        "treated": "arm > 0",
        "recency_days": "recency (months since last purchase) x 30.4375",
        "monetary_12m": "history (dollars spent in the past year)",
        "value_h": "spend (dollars in the two weeks after the send)",
    }
    derivations: dict[str, str] = {
        "customer_id": "row position, zero-padded; stable for a given source ordering",
        "period": "constant 0; the experiment has a single outcome window",
        "arm": "PRISM's control-is-0 convention applied to the three e-mail segments",
        "recency_days": f"months x {_DAYS_PER_MONTH} calendar days",
        "value_h": "realised, undiscounted two-week revenue; NOT a 12-month discounted CLV",
    }

    cat_cols: list[str] = []
    if "mens" in df.columns and "womens" in df.columns:
        mens = pd.to_numeric(df["mens"], errors="coerce").astype("float64")
        womens = pd.to_numeric(df["womens"], errors="coerce").astype("float64")
        panel["n_categories_12m"] = mens + womens
        panel["basket_diversity"] = (mens + womens) / 2.0
        mapped["n_categories_12m"] = "mens + womens (product categories bought in the past year)"
        mapped["basket_diversity"] = "(mens + womens) / 2 available categories"
        derivations["n_categories_12m"] = "the source only distinguishes two merchandise categories"
        derivations["basket_diversity"] = "share of the two-category catalogue purchased from"
        cat_cols += ["mens", "womens"]

    keep = [c for c in ("recency", "history_segment", "history", "mens", "womens", "zip_code",
                        "newbie", "channel", "segment", "visit", "conversion", "spend") if c in df.columns]
    extras = df[keep].copy()
    extras["arm_name"] = segment
    extras["propensity_known"] = 1.0 / 3.0

    notes = [
        "This is the one genuinely randomised source: assignment is uniform over three arms, so "
        "the true propensity is 1/3 (src_propensity_known) and no propensity model is required.",
        "Outcomes are visit / conversion / spend over two weeks. There is no churn event and no "
        "follow-up time, so churn_next, event_time and event_observed stay NaN.",
        "offer_cost is NaN: the cost of sending an e-mail is not recorded in the source. Attach "
        "your own cost assumption before running prism.decision.optimize on this panel.",
        "src_channel (Phone/Web/Multichannel) and src_zip_code (Rural/Surburban/Urban) do not fit "
        "the canonical level sets for channel/region, so they are kept unmapped rather than recoded.",
        "value_h holds the post-treatment outcome (two-week spend). It is an outcome, never a "
        "feature: attrs['leaky_features']['value_h'] is empty only because none of the 23 canonical "
        "features are post-treatment here.",
    ]
    leaky: dict[str, list[str]] = {"value_h": []}
    return _finalise(panel, extras, name="hillstrom", mapped=mapped, derivations=derivations,
                     notes=notes, source_attrs=df.attrs, leaky_features=leaky)


def _map_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Online Retail II invoice lines into a one-row-per-customer panel.

    **Point-in-time discipline.** The source has no churn label, so one has to be
    constructed -- and the obvious construction ("no purchase in the last 90 days")
    silently leaks, because it is a deterministic threshold of ``recency_days``, which is
    itself one of the 23 canonical features. A model trained on such a panel scores a
    perfect AUC and teaches you nothing.

    This adapter therefore splits the history at a decision date::

        as_of_date = max(InvoiceDate) - RETAIL_LAPSE_DAYS

    *Every* feature is computed strictly from invoices at or before ``as_of_date``, and the
    label is whether the customer bought at all in the ``RETAIL_LAPSE_DAYS`` that follow
    it. Feature window and label window are disjoint, so ``churn_next`` is genuinely a
    forward-looking outcome and the SPEC rule "any feature at period ``t`` may only use
    information with ``as_of_date <= panel.as_of_date[t]``" holds literally.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw invoice lines as returned by :func:`load_real`.

    Returns
    -------
    pandas.DataFrame
        One row per customer active on or before the decision date, canonical panel
        columns plus ``src_*`` passthroughs.
    """
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    need = ["invoice", "stockcode", "quantity", "invoicedate", "price", "customer_id"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(
            f"frame does not look like the 'online_retail_ii' source: missing {missing}; "
            f"got {list(df.columns)[:12]}"
        )

    tx = pd.DataFrame(
        {
            "customer_id": df[cols["customer_id"]],
            "invoice": df[cols["invoice"]].astype("string"),
            "stockcode": df[cols["stockcode"]].astype("string"),
            "quantity": pd.to_numeric(df[cols["quantity"]], errors="coerce"),
            "price": pd.to_numeric(df[cols["price"]], errors="coerce"),
            "invoice_date": pd.to_datetime(df[cols["invoicedate"]], errors="coerce"),
            "country": df[cols["country"]].astype("string") if "country" in cols else pd.NA,
        }
    )
    n_raw = len(tx)
    tx = tx[tx["customer_id"].notna() & tx["invoice_date"].notna()]
    tx = tx[(tx["quantity"] > 0) & (tx["price"] > 0)].copy()
    if tx.empty:
        raise ValueError("no usable invoice lines after dropping returns and guest checkouts")
    tx["revenue"] = tx["quantity"] * tx["price"]
    tx["customer_id"] = tx["customer_id"].astype("float64").astype("int64").astype("string")

    # ---- temporal split: features <= cutoff, label strictly after it -------------------
    data_end = tx["invoice_date"].max()
    cutoff = data_end - pd.Timedelta(days=int(RETAIL_LAPSE_DAYS))
    hist = tx[tx["invoice_date"] <= cutoff]
    future = tx[tx["invoice_date"] > cutoff]
    if hist.empty:
        raise ValueError(
            f"no invoice lines on or before the decision date {cutoff:%Y-%m-%d}; the source span "
            f"is shorter than RETAIL_LAPSE_DAYS={RETAIL_LAPSE_DAYS}"
        )

    window_start = cutoff - pd.Timedelta(days=365)
    recent = hist[hist["invoice_date"] > window_start]

    lifetime = hist.groupby("customer_id", sort=True).agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        n_invoices_life=("invoice", "nunique"),
        revenue_life=("revenue", "sum"),
    )
    r12 = recent.groupby("customer_id", sort=True).agg(
        frequency_12m=("invoice", "nunique"),
        monetary_12m=("revenue", "sum"),
        n_categories_12m=("stockcode", "nunique"),
        n_lines_12m=("stockcode", "size"),
    )
    country = hist.sort_values("invoice_date").groupby("customer_id", sort=True)["country"].last()
    agg = lifetime.join(r12, how="left").join(country.rename("country"), how="left")
    agg[["frequency_12m", "monetary_12m", "n_categories_12m", "n_lines_12m"]] = agg[
        ["frequency_12m", "monetary_12m", "n_categories_12m", "n_lines_12m"]
    ].fillna(0.0)

    # ---- the held-out label window (no feature may touch it) ---------------------------
    fwd = future.groupby("customer_id", sort=True).agg(
        n_invoices_next=("invoice", "nunique"),
        revenue_next=("revenue", "sum"),
    )
    fwd = fwd.reindex(agg.index).fillna(0.0)
    retained = fwd["n_invoices_next"].to_numpy(dtype="float64") > 0.0
    lapsed = (~retained).astype("float64")

    n = len(agg)
    panel = _blank_panel(n)
    panel["customer_id"] = pd.Series(agg.index.to_numpy(), index=panel.index, dtype="string")
    panel["period"] = 0.0
    panel["as_of_date"] = pd.Series(np.repeat(cutoff, n), index=panel.index)
    panel["cohort"] = pd.Series(agg["first_purchase"].dt.strftime("%Y-%m").to_numpy(), index=panel.index)

    recency_days = (cutoff - agg["last_purchase"]).dt.total_seconds().to_numpy() / 86400.0
    tenure_months = (cutoff - agg["first_purchase"]).dt.total_seconds().to_numpy() / (86400.0 * _DAYS_PER_MONTH)
    freq = agg["frequency_12m"].to_numpy(dtype="float64")
    monetary = agg["monetary_12m"].to_numpy(dtype="float64")
    n_lines = agg["n_lines_12m"].to_numpy(dtype="float64")
    n_cat = agg["n_categories_12m"].to_numpy(dtype="float64")

    panel["recency_days"] = recency_days
    panel["tenure_months"] = tenure_months
    panel["frequency_12m"] = freq
    panel["monetary_12m"] = monetary
    panel["avg_order_value"] = np.divide(monetary, freq, out=np.full(n, np.nan), where=freq > 0)
    panel["n_categories_12m"] = n_cat
    panel["basket_diversity"] = np.divide(n_cat, n_lines, out=np.full(n, np.nan), where=n_lines > 0)

    # Survival framing. A lapse is *declared* RETAIL_LAPSE_DAYS after the customer's last
    # purchase -- the moment the evidence for it is complete -- not at the purchase itself.
    # Dating it at the purchase gave every one-order customer event_time == 0 with
    # event_observed == 1: an instantaneous event that breaks Cox risk sets and the
    # discrete-time hazard person-period expansion alike.
    grace_months = float(RETAIL_LAPSE_DAYS) / _DAYS_PER_MONTH
    months_first_to_last = (agg["last_purchase"] - agg["first_purchase"]).dt.total_seconds().to_numpy() / (
        86400.0 * _DAYS_PER_MONTH
    )
    months_first_to_end = (data_end - agg["first_purchase"]).dt.total_seconds().to_numpy() / (
        86400.0 * _DAYS_PER_MONTH
    )
    panel["churn_next"] = lapsed
    panel["event_observed"] = lapsed
    panel["event_time"] = np.where(retained, months_first_to_end, months_first_to_last + grace_months)
    panel["value_h"] = fwd["revenue_next"].to_numpy(dtype="float64")

    mapped: dict[str, str] = {
        "customer_id": "Customer ID",
        "period": "single decision date, so every row is period 0",
        "as_of_date": f"decision date = max(InvoiceDate) - {RETAIL_LAPSE_DAYS}d = {cutoff:%Y-%m-%d}",
        "cohort": "first purchase month, YYYY-MM",
        "recency_days": "days from the last invoice at or before as_of_date to as_of_date",
        "tenure_months": "months from the first invoice to as_of_date",
        "frequency_12m": "distinct invoices in the 365 days before as_of_date",
        "monetary_12m": "sum(Quantity x Price) in the 365 days before as_of_date",
        "avg_order_value": "monetary_12m / frequency_12m",
        "n_categories_12m": "distinct StockCodes in the trailing 12 months",
        "basket_diversity": "distinct StockCodes / invoice lines in the trailing 12 months",
        "churn_next": f"no purchase in the {RETAIL_LAPSE_DAYS} days AFTER as_of_date (held-out window)",
        "event_time": "months from the first purchase to the declared lapse, else to the end of data",
        "event_observed": "same held-out lapse label",
        "value_h": f"realised revenue in the {RETAIL_LAPSE_DAYS} days after as_of_date (undiscounted)",
    }
    derivations: dict[str, str] = {
        "as_of_date": (
            f"the transaction log is split at max(InvoiceDate) - {RETAIL_LAPSE_DAYS} days. Features use "
            "only invoices at or before this date; the label uses only invoices after it, so the two "
            "windows are disjoint and recency_days cannot determine churn_next."
        ),
        "avg_order_value": "NaN for customers with no invoice in the trailing 12-month window",
        "churn_next": (
            f"CONVENTION, not a fact in the source: a customer with no order in the {RETAIL_LAPSE_DAYS} "
            "days following as_of_date is labelled lapsed. Change RETAIL_LAPSE_DAYS to test sensitivity "
            "(it moves both the decision date and the label window)."
        ),
        "event_observed": "the same held-out inactivity rule, so it is a label, not an observed cancellation",
        "event_time": (
            f"lapsed: months(first -> last purchase) + {RETAIL_LAPSE_DAYS} days of grace, i.e. the date "
            "the lapse is declared; retained: months(first purchase -> end of data), right-censored. "
            "Both branches are strictly positive by construction."
        ),
        "value_h": (
            f"sum of revenue in the held-out {RETAIL_LAPSE_DAYS}-day window; undiscounted and over a "
            f"{RETAIL_LAPSE_DAYS}-day horizon, NOT the 12-month discounted CLV the simulator puts here."
        ),
    }

    extras = pd.DataFrame(
        {
            "country": agg["country"].to_numpy(),
            "n_invoices_lifetime": agg["n_invoices_life"].to_numpy(dtype="float64"),
            "revenue_lifetime": agg["revenue_life"].to_numpy(dtype="float64"),
            "first_purchase": agg["first_purchase"].to_numpy(),
            "last_purchase": agg["last_purchase"].to_numpy(),
            "decision_date": np.repeat(cutoff, n),
            "data_end_date": np.repeat(data_end, n),
            "n_lines_12m": n_lines,
            "n_invoices_next": fwd["n_invoices_next"].to_numpy(dtype="float64"),
            "revenue_next": fwd["revenue_next"].to_numpy(dtype="float64"),
        }
    )

    notes = [
        f"Aggregated {n_raw:,} invoice lines into {n:,} customers active on or before the decision "
        f"date {cutoff:%Y-%m-%d} (the data itself ends {data_end:%Y-%m-%d}).",
        "Returns (Quantity <= 0), zero-price lines and guest checkouts (no Customer ID) are dropped.",
        f"Feature window and label window are disjoint: features use invoices <= {cutoff:%Y-%m-%d}, "
        f"churn_next and value_h use only the {RETAIL_LAPSE_DAYS} days after it. Labelling on trailing "
        "recency instead would make churn_next a deterministic function of the recency_days feature "
        "(single-feature AUC 1.0), which is why it is not done that way here.",
        "Customers whose first ever order falls inside the held-out window are excluded: at the "
        "decision date they do not exist yet, so scoring them would itself be leakage.",
        "No experiment exists here, so arm, treated and offer_cost are NaN; the value of this source "
        "is RFM/CLV and point-in-time feature engineering, not uplift.",
        f"churn_next and event_observed come from a {RETAIL_LAPSE_DAYS}-day inactivity convention, "
        "not from an observed cancellation.",
        "src_country is kept raw: the canonical region levels (north/south/east/west) do not apply.",
    ]
    return _finalise(
        panel,
        extras,
        name="online_retail_ii",
        mapped=mapped,
        derivations=derivations,
        notes=notes,
        source_attrs=df.attrs,
        leaky_features={},
    )


_MAPPERS = {
    "telco_churn": _map_telco,
    "online_retail_ii": _map_online_retail,
    "hillstrom": _map_hillstrom,
}


def map_to_panel(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Translate a real source frame into the canonical PRISM gold-panel layout.

    The output always carries every column in :data:`prism.data.schema.PANEL_COLUMNS`, in
    order. Columns the source genuinely cannot supply are ``NaN`` (or ``NaT``/``None``) and
    are listed in ``df.attrs["unmapped_columns"]`` -- nothing is invented to fill them.
    Source columns worth keeping are appended with an ``src_`` prefix.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame returned by :func:`load_real` (or an equivalent copy read from disk).
    name : str
        Dataset key: ``"telco_churn"``, ``"online_retail_ii"`` or ``"hillstrom"``.

    Returns
    -------
    pandas.DataFrame
        The adapted panel. ``df.attrs`` holds ``unmapped_columns``, ``mapped_columns``,
        ``derivations``, ``extra_columns``, ``notes``, ``citation`` and ``license``.

    Raises
    ------
    ValueError
        If ``name`` is unknown, ``df`` is empty, or ``df`` lacks the source's key columns.
        Unlike :func:`load_real` this is a programming error, not a network condition, so
        it is surfaced rather than swallowed.

    Examples
    --------
    >>> panel = map_to_panel(load_real("hillstrom"), "hillstrom")     # doctest: +SKIP
    >>> panel["arm"].value_counts().to_dict()                         # doctest: +SKIP
    {0.0: 21306, 2.0: 21307, 1.0: 21387}
    """
    if name not in _MAPPERS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(AVAILABLE_DATASETS)}")
    if df is None or len(df) == 0:
        raise ValueError(f"cannot map an empty frame for dataset {name!r}")
    return _MAPPERS[name](df)


def load_real_panel(
    name: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Convenience wrapper: :func:`load_real` followed by :func:`map_to_panel`.

    Parameters
    ----------
    name : str
        Dataset key.
    cache_dir : str or pathlib.Path, default ``"artifacts/data/real"``
        Cache directory.
    **kwargs
        Forwarded to :func:`load_real` (``timeout``, ``force_refresh``, ``sample_rows``...).

    Returns
    -------
    pandas.DataFrame or None
        The mapped panel, or ``None`` if the source could not be loaded or mapped. Like
        :func:`load_real`, this never raises.
    """
    raw = load_real(name, cache_dir, **kwargs)
    if raw is None:
        return None
    try:
        return map_to_panel(raw, name)
    except Exception as exc:
        log.warning("%s: mapping to the canonical panel failed (%s): %s", name, type(exc).__name__, exc)
        return None


# ======================================================================================
# Public API: introspection
# ======================================================================================
def describe_datasets() -> pd.DataFrame:
    """Summarise :data:`AVAILABLE_DATASETS` as a tidy frame.

    Returns
    -------
    pandas.DataFrame
        One row per dataset with columns ``name``, ``title``, ``entity``, ``randomised``,
        ``n_rows_approx``, ``n_cols_approx``, ``size_mb_approx``, ``format``,
        ``demonstrates``, ``description``, ``license``, ``citation``, ``url``,
        ``n_mirrors``, ``requires``, ``suggested_timeout`` and ``download_hint``.
    """
    rows: list[dict[str, Any]] = []
    for name, spec in AVAILABLE_DATASETS.items():
        rows.append(
            {
                "name": name,
                "title": spec["title"],
                "entity": spec["entity"],
                "randomised": bool(spec["randomised"]),
                "n_rows_approx": int(spec["n_rows_approx"]),
                "n_cols_approx": int(spec["n_cols_approx"]),
                "size_mb_approx": float(spec["size_mb_approx"]),
                "format": spec["format"],
                "demonstrates": spec["demonstrates"],
                "description": spec["description"],
                "license": spec["license"],
                "citation": spec["citation"],
                "url": spec["url"],
                "n_mirrors": len(spec["urls"]),
                "requires": ", ".join(spec["requires"]) or "-",
                "suggested_timeout": float(spec["suggested_timeout"]),
                "download_hint": spec["download_hint"],
            }
        )
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def dataset_status(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Report which datasets are already cached locally, and how big the cache is.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, default ``"artifacts/data/real"``
        Directory to inspect. It is not created by this call.

    Returns
    -------
    pandas.DataFrame
        One row per dataset: ``name``, ``cached`` (parquet present), ``raw_cached``,
        ``raw_mb``, ``parquet_mb``, ``n_rows``, ``n_cols``, ``fetched_at``, ``sha256``
        (first 12 chars) and ``parquet_path``.
    """
    rows: list[dict[str, Any]] = []
    for name in AVAILABLE_DATASETS:
        paths = _cache_paths(name, cache_dir)
        meta = _read_meta(paths["meta"])
        raw_exists = paths["raw"].exists()
        pq_exists = paths["parquet"].exists()
        rows.append(
            {
                "name": name,
                "cached": bool(pq_exists),
                "raw_cached": bool(raw_exists),
                "raw_mb": round(paths["raw"].stat().st_size / 1e6, 2) if raw_exists else np.nan,
                "parquet_mb": round(paths["parquet"].stat().st_size / 1e6, 2) if pq_exists else np.nan,
                "n_rows": int(meta["n_rows"]) if isinstance(meta.get("n_rows"), int) else np.nan,
                "n_cols": int(meta["n_cols"]) if isinstance(meta.get("n_cols"), int) else np.nan,
                "fetched_at": meta.get("fetched_at", "-"),
                "sha256": str(meta.get("sha256", ""))[:12] or "-",
                "parquet_path": str(paths["parquet"]),
            }
        )
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def clear_cache(name: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[str]:
    """Delete cached files for one dataset (or all of them).

    Parameters
    ----------
    name : str, optional
        Dataset key. ``None`` clears every known dataset.
    cache_dir : str or pathlib.Path, default ``"artifacts/data/real"``
        Cache directory.

    Returns
    -------
    list of str
        Paths actually removed.
    """
    targets = [name] if name is not None else list(AVAILABLE_DATASETS)
    removed: list[str] = []
    for key in targets:
        if key not in AVAILABLE_DATASETS:
            log.warning("clear_cache: unknown dataset %r, skipped", key)
            continue
        for path in _cache_paths(key, cache_dir).values():
            if path.is_file():
                try:
                    path.unlink()
                    removed.append(str(path))
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    log.warning("could not delete %s: %s", path, exc)
    return removed


# ======================================================================================
# Smoke test
# ======================================================================================
class _Checks:
    """Tiny assertion recorder so the smoke test reports *all* failures, not just the first."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def __call__(self, ok: bool, label: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), label, detail))
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        return bool(ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for ok, _, _ in self.rows if not ok)

    @property
    def n_total(self) -> int:
        return len(self.rows)


def _check_panel_invariants(panel: pd.DataFrame, name: str, chk: _Checks) -> None:
    """Assert the invariants every adapted panel must satisfy, whatever the source.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of :func:`map_to_panel`.
    name : str
        Dataset key, used in the assertion labels.
    chk : _Checks
        Recorder to accumulate results into.
    """
    cols = list(panel.columns)
    chk(cols[: len(PANEL_COLUMNS)] == list(PANEL_COLUMNS), f"{name}: canonical columns first, in order")
    chk(all(str(c).startswith("src_") for c in cols[len(PANEL_COLUMNS):]),
        f"{name}: every extra column carries the src_ prefix")
    chk(len(panel) > 0, f"{name}: non-empty", f"{len(panel)} rows")

    for key in ("unmapped_columns", "mapped_columns", "derivations", "notes", "citation",
                "license", "leaky_features"):
        chk(key in panel.attrs, f"{name}: attrs['{key}'] present")
    mapped = set(panel.attrs.get("mapped_columns", {}))
    unmapped = set(panel.attrs.get("unmapped_columns", []))
    chk(mapped.isdisjoint(unmapped) and mapped | unmapped == set(PANEL_COLUMNS),
        f"{name}: mapped + unmapped partition PANEL_COLUMNS",
        f"{len(mapped)} mapped, {len(unmapped)} unmapped")
    # A column claimed as mapped must actually hold data; an unmapped one must be all-missing.
    empty_claimed = [c for c in mapped if panel[c].isna().all()]
    filled_unclaimed = [c for c in unmapped if panel[c].notna().any()]
    chk(not empty_claimed, f"{name}: no column claimed mapped is entirely missing", str(empty_claimed[:4]))
    chk(not filled_unclaimed, f"{name}: no column claimed unmapped carries data", str(filled_unclaimed[:4]))

    # Identity.
    cid = panel["customer_id"]
    chk(cid.notna().all() and not cid.duplicated().any(), f"{name}: customer_id non-null and unique")

    # Binary / categorical outcome ranges.
    for col in ("churn_next", "event_observed", "treated"):
        v = pd.to_numeric(panel[col], errors="coerce").dropna()
        if len(v):
            chk(set(v.unique()) <= {0, 1}, f"{name}: {col} in {{0, 1}}", f"observed {sorted(set(v.unique()))}")
    arm = pd.to_numeric(panel["arm"], errors="coerce").dropna()
    if len(arm):
        chk(bool((arm >= 0).all()) and bool((arm == arm.round()).all()),
            f"{name}: arm is a non-negative integer index", f"max {arm.max():.0f}")
        tre = pd.to_numeric(panel["treated"], errors="coerce")
        chk(bool(((arm > 0).astype(float) == tre.loc[arm.index]).all()), f"{name}: treated == (arm > 0)")

    # Survival columns must be usable by prism.models.survival.
    et = pd.to_numeric(panel["event_time"], errors="coerce")
    eo = pd.to_numeric(panel["event_observed"], errors="coerce")
    if et.notna().any():
        chk(bool((et.dropna() >= 0).all()), f"{name}: event_time non-negative", f"min {et.min():.4f}")
        bad = int(((et == 0) & (eo == 1)).sum())
        chk(bad == 0, f"{name}: no observed event at time 0", f"{bad} degenerate rows")
    if et.notna().any() and eo.notna().any():
        chk(bool(eo.notna().eq(et.notna()).all()), f"{name}: event_time and event_observed co-defined")

    # Point-in-time safety: no feature may be dated after the decision date.
    aod = panel["as_of_date"]
    if aod.notna().any():
        for extra in ("src_last_purchase", "src_first_purchase"):
            if extra in panel.columns:
                ts = pd.to_datetime(panel[extra], errors="coerce")
                chk(bool((ts <= aod).all()), f"{name}: {extra} <= as_of_date (no future information)")

    # The leakage screen that caught the original online_retail_ii bug: no single canonical
    # feature may separate the label perfectly.
    y = pd.to_numeric(panel["churn_next"], errors="coerce")
    declared = set()
    for v in panel.attrs.get("leaky_features", {}).values():
        declared |= set(v)
    if y.notna().any() and y.nunique() > 1:
        worst_name, worst_auc = "-", 0.5
        for feat in NUMERIC_FEATURES:
            if feat in declared:
                continue
            auc = _single_feature_auc(pd.to_numeric(panel[feat], errors="coerce"), y)
            if auc is not None and abs(auc - 0.5) > abs(worst_auc - 0.5):
                worst_name, worst_auc = feat, auc
        chk(abs(worst_auc - 0.5) < 0.499,
            f"{name}: no undeclared feature determines churn_next",
            f"worst is {worst_name} at AUC {worst_auc:.4f}")


def _single_feature_auc(x: pd.Series, y: pd.Series) -> float | None:
    """Rank (Mann-Whitney) AUC of one feature against a binary label, or None if undefined."""
    m = x.notna() & y.notna()
    if int(m.sum()) < 20:
        return None
    xv = x[m].to_numpy(dtype="float64")
    yv = y[m].to_numpy(dtype="float64")
    n1 = float(yv.sum())
    n0 = float(len(yv) - n1)
    if n1 == 0.0 or n0 == 0.0 or len(np.unique(xv)) < 2:
        return None
    ranks = pd.Series(xv).rank(method="average").to_numpy()
    return float((ranks[yv == 1.0].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))


def _smoke() -> int:
    """Run the module smoke test. Returns a process exit code (0 = every assertion held)."""
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)
    t_start = time.perf_counter()
    chk = _Checks()

    print("=" * 96)
    print("PRISM real-dataset adapters")
    print("=" * 96)

    desc = describe_datasets()
    print("\n[1] describe_datasets()")
    print(desc[["name", "entity", "randomised", "n_rows_approx", "size_mb_approx", "format", "requires"]].to_string(index=False))
    chk(len(desc) == len(AVAILABLE_DATASETS) and desc["name"].is_unique, "registry: one unique row per dataset")
    chk(int(desc["randomised"].sum()) == 1, "registry: exactly one randomised source (hillstrom) for uplift")
    chk(all(name in _MAPPERS for name in AVAILABLE_DATASETS), "registry: every dataset has a mapper")

    print("\n[2] dataset_status()")
    status = dataset_status()
    print(status[["name", "cached", "raw_cached", "raw_mb", "n_rows", "n_cols", "fetched_at"]].to_string(index=False))
    with tempfile.TemporaryDirectory(prefix="prism_real_smoke_") as tmp:
        cold = str(Path(tmp) / "cold")
        chk(len(dataset_status(cold)) == len(AVAILABLE_DATASETS),
            "dataset_status on a missing directory still returns a full frame")

        print("\n[3] error and offline paths (nothing here may raise)")
        chk(load_real("not_a_dataset") is None, "load_real on an unknown name returns None")
        chk(load_real("telco_churn", cache_dir=cold, allow_download=False) is None,
            "load_real with an empty cache and allow_download=False returns None")
        chk(load_real_panel("not_a_dataset", cache_dir=cold) is None,
            "load_real_panel on an unknown name returns None")
        chk(load_real_panel("telco_churn", cache_dir=cold, allow_download=False) is None,
            "load_real_panel degrades to None instead of raising when nothing is cached")
    for bad, label in ((pd.DataFrame({"nope": [1, 2]}), "a frame with the wrong columns"),
                       (pd.DataFrame(), "an empty frame")):
        try:
            map_to_panel(bad, "telco_churn")
            chk(False, f"map_to_panel raises ValueError on {label}")
        except ValueError:
            chk(True, f"map_to_panel raises ValueError on {label}")
        except Exception as exc:  # noqa: BLE001 - the point is that it must be a ValueError
            chk(False, f"map_to_panel raises ValueError on {label}", f"got {type(exc).__name__}")

    print("\n[4] canonical panel target")
    print(f"  PANEL_COLUMNS = {len(PANEL_COLUMNS)}  (features: {len(FEATURES)} = "
          f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical)")

    print("\n[5] adapting every cached source (downloading telco if the cache is cold)")
    n_adapted = 0
    for name in sorted(AVAILABLE_DATASETS):
        budget = 10.0 if name == "telco_churn" else 0.0
        raw = load_real(name, timeout=budget or 1.0, max_seconds=budget or 1.0,
                        allow_download=budget > 0.0)
        if raw is None:
            print(f"  - {name:16s} not cached and not fetched in budget -- skipped")
            continue
        fingerprint = pd.util.hash_pandas_object(raw, index=False).sum()
        panel = map_to_panel(raw, name)
        chk(pd.util.hash_pandas_object(raw, index=False).sum() == fingerprint,
            f"{name}: map_to_panel does not mutate the caller's frame")
        print(f"  - {name:16s} raw {raw.shape} -> panel {panel.shape}, "
              f"{len(panel.attrs['mapped_columns'])}/{len(PANEL_COLUMNS)} canonical columns populated")
        _check_panel_invariants(panel, name, chk)
        n_adapted += 1

        if name == "telco_churn":
            churn = pd.to_numeric(panel["churn_next"], errors="coerce")
            chk(0.20 < float(churn.mean()) < 0.32, "telco_churn: churn rate matches the published 26.5%",
                f"{churn.mean():.4f}")
            chk(panel["contract_type"].dropna().isin(["monthly", "annual"]).all(),
                "telco_churn: contract_type uses canonical levels")
        if name == "hillstrom":
            counts = panel["arm"].value_counts()
            share = counts / counts.sum()
            chk(set(counts.index) == {0, 1, 2} and bool(((share - 1 / 3).abs() < 0.01).all()),
                "hillstrom: three arms, each randomised to about 1/3", share.round(4).to_dict())
            by_arm = panel.groupby("arm")["value_h"].mean()
            chk(by_arm.loc[1] > by_arm.loc[0] and by_arm.loc[2] > by_arm.loc[1],
                "hillstrom: mean spend reproduces control < womens < mens",
                by_arm.round(3).to_dict())
        if name == "online_retail_ii":
            aod = panel["as_of_date"].iloc[0]
            end = pd.to_datetime(panel["src_data_end_date"].iloc[0])
            chk((end - aod).days == RETAIL_LAPSE_DAYS,
                "online_retail_ii: decision date sits one lapse window before the end of data",
                f"{aod:%Y-%m-%d} -> {end:%Y-%m-%d}")
            rec = pd.to_numeric(panel["recency_days"], errors="coerce")
            lab = pd.to_numeric(panel["churn_next"], errors="coerce")
            chk(not bool(((rec > RETAIL_LAPSE_DAYS).astype(float) == lab).all()),
                "online_retail_ii: churn_next is NOT a threshold of recency_days (the old leak)")
            chk(bool((rec >= 0).all()), "online_retail_ii: recency measured backwards from as_of_date")
            fut = pd.to_numeric(panel["src_n_invoices_next"], errors="coerce")
            chk(bool(((fut == 0).astype(float) == lab).all()),
                "online_retail_ii: the label is exactly 'no order in the held-out window'")

    chk(n_adapted >= 1, "at least one source was adapted end to end", f"{n_adapted}/3")

    print("\n[6] determinism of the subsample path")
    a = load_real("telco_churn", allow_download=False, sample_rows=250, random_state=7)
    if a is None:
        print("  telco not cached -- skipped")
    else:
        b = load_real("telco_churn", allow_download=False, sample_rows=250, random_state=7)
        c = load_real("telco_churn", allow_download=False, sample_rows=250, random_state=8)
        chk(a.equals(b), "sample_rows with the same random_state is bit-identical")
        chk(not a.equals(c), "a different random_state gives a different sample")
        chk(len(a) == 250, "sample_rows honoured", f"{len(a)} rows")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 96)
    print(f"{chk.n_total - chk.n_failed}/{chk.n_total} assertions passed in {elapsed:.1f}s")
    if chk.n_failed:
        for ok, label, detail in chk.rows:
            if not ok:
                print(f"  FAILED: {label}  {detail}")
        print("real.py FAILED")
        return 1
    print("real.py OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - smoke test
    raise SystemExit(_smoke())
