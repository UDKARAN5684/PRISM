"""Medallion (bronze / silver / gold) warehouse for PRISM, DuckDB-backed.

Why this module exists
----------------------
The PRISM pipeline moves a frame through three curation layers:

``bronze``
    Raw, as-landed source data (the simulated event log, the customer dimension,
    or a real-world extract). Nothing is repaired here; bronze is the audit trail.
``silver``
    Conformed and cleaned -- deduplicated, schema-repaired, dtypes enforced. This is
    the output of :mod:`prism.data.messy`'s ``clean_panel``.
``gold``
    Analysis-ready. The contract-validated panel that models and the causal layer read.

Physically a table lives at ``"<layer>_<name>"`` (e.g. ``gold_panel``), so the layer is
visible in every query and a ``SELECT`` can never silently read the wrong curation stage.

Backends
--------
DuckDB is used when it is importable (an embedded OLAP engine: zero-config, one file,
full SQL). When it is absent the class falls back **transparently** to one parquet file
per table in a sibling directory, so the rest of PRISM never branches on availability.
The two backends differ in exactly one visible way -- see :meth:`Warehouse.sql`.

Round-trip fidelity
-------------------
Neither backend preserves pandas extension dtypes by itself: DuckDB maps the pandas
``string`` dtype to ``VARCHAR`` and hands it back as ``object``. So every write records
the frame's dtype map as JSON in the lineage table, and :meth:`Warehouse.read` replays it.
A frame of ``string`` / ``int64`` / ``float64`` / ``datetime64[ns]`` columns therefore
round-trips with dtypes *and* values intact on both backends (asserted in the smoke test).
The DataFrame index is **not** preserved: frames are stored positionally and come back
with a fresh ``RangeIndex``.

Content hashing
---------------
``sha256`` in the lineage is not a hash of the file bytes (those differ run to run because
of compression metadata and row-group layout). It is a hash of the *content*:
:func:`pandas.util.hash_pandas_object` reduces every row to a ``uint64``, and the sha256 is
taken over the column names, the dtype strings and those row hashes. That is cheap
(vectorised, one pass, no serialisation) and stable for a given frame on a given platform
and pandas version -- enough to answer "did the gold panel actually change?".
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd

from prism.utils.logging import get_logger
from prism.utils.optional import HAS_DUCKDB, require

__all__ = [
    "LAYERS",
    "DEFAULT_WAREHOUSE_PATH",
    "Warehouse",
    "build_medallion",
    "frame_sha256",
]

LOG = get_logger("data.warehouse")

#: The three medallion curation layers, in promotion order.
LAYERS: tuple[str, ...] = ("bronze", "silver", "gold")

#: Default on-disk location, matching ``PrismConfig.paths.warehouse``.
DEFAULT_WAREHOUSE_PATH: str = "artifacts/warehouse.duckdb"

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_LINEAGE_TABLE: str = "_prism_lineage"
_RESERVED_PREFIX: str = "_prism"
_STAGE_VIEW: str = "_prism_stage"

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SELECT_STAR_RE = re.compile(
    r"""^\s*select\s+\*\s+from\s+["'`\[]?([A-Za-z_][A-Za-z0-9_]*)["'`\]]?\s*;?\s*$""",
    re.IGNORECASE,
)

_LINEAGE_COLUMNS: tuple[str, ...] = ("table", "layer", "n_rows", "n_cols", "written_at", "sha256")
_LINEAGE_STORE_COLUMNS: tuple[str, ...] = (
    "write_id",
    "table",
    "layer",
    "name",
    "n_rows",
    "n_cols",
    "written_at",
    "sha256",
    "dtypes",
)
_TABLE_COLUMNS: tuple[str, ...] = ("table", "layer", "name", "n_rows", "n_cols")

_EMPTY_LINEAGE = pd.DataFrame(
    {
        "write_id": pd.Series([], dtype="int64"),
        "table": pd.Series([], dtype="object"),
        "layer": pd.Series([], dtype="object"),
        "name": pd.Series([], dtype="object"),
        "n_rows": pd.Series([], dtype="int64"),
        "n_cols": pd.Series([], dtype="int64"),
        "written_at": pd.Series([], dtype="object"),
        "sha256": pd.Series([], dtype="object"),
        "dtypes": pd.Series([], dtype="object"),
    }
)


# ======================================================================================
# module-level helpers
# ======================================================================================
def _resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` to an absolute :class:`~pathlib.Path`, Windows-safely.

    Relative paths are interpreted against the PRISM project root rather than the
    current working directory, so a warehouse opened from a notebook, a test or the
    CLI all land on the same file.

    Parameters
    ----------
    path : str or pathlib.Path
        Absolute or project-relative location.

    Returns
    -------
    pathlib.Path
        Absolute path. No filesystem access is performed.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    # ``Path.resolve`` on Windows normalises separators and drive casing without
    # requiring the file to exist (strict=False is the default on 3.6+).
    return Path(p).resolve()


def _has_parquet_store(parquet_dir: Path) -> bool:
    """Return whether a parquet-fallback sidecar directory holds any table.

    Parameters
    ----------
    parquet_dir : pathlib.Path
        Candidate ``<stem>_parquet`` directory.

    Returns
    -------
    bool
        True if the directory exists and contains at least one ``.parquet`` file.
    """
    return parquet_dir.is_dir() and any(parquet_dir.glob("*.parquet"))


def _validate_layer(layer: str) -> str:
    """Return the canonical layer name or raise.

    Parameters
    ----------
    layer : str
        Candidate layer.

    Returns
    -------
    str
        One of :data:`LAYERS`.

    Raises
    ------
    ValueError
        If ``layer`` is not a medallion layer.
    """
    if not isinstance(layer, str):
        raise ValueError(f"layer must be a string, got {type(layer).__name__}")
    key = layer.strip().lower()
    if key not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
    return key


def _validate_name(name: str) -> str:
    """Return a validated table name.

    Names must be simple SQL/filesystem identifiers. This is what keeps
    :meth:`Warehouse.write` free of SQL-injection and path-traversal risk, since the
    name is interpolated into DDL and into a file path.

    Parameters
    ----------
    name : str
        Candidate dataset name, e.g. ``"panel"``.

    Returns
    -------
    str
        The lower-cased name.

    Raises
    ------
    ValueError
        If the name is not ``[A-Za-z][A-Za-z0-9_]*`` or collides with the reserved
        ``_prism`` internal namespace.
    """
    if not isinstance(name, str):
        raise ValueError(f"name must be a string, got {type(name).__name__}")
    key = name.strip().lower()
    if not _NAME_RE.match(key):
        raise ValueError(
            f"name must match [A-Za-z][A-Za-z0-9_]* (letters, digits, underscore; "
            f"no dots, spaces, quotes or path separators), got {name!r}"
        )
    if key.startswith(_RESERVED_PREFIX):
        raise ValueError(f"names starting with {_RESERVED_PREFIX!r} are reserved for warehouse internals")
    return key


def _q(identifier: str) -> str:
    """Quote a SQL identifier for DuckDB (doubling embedded quotes)."""
    return '"' + identifier.replace('"', '""') + '"'


def frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame: sha256 over column names, dtypes and row hashes.

    The hash is order-sensitive in both axes and ignores the index, mirroring how the
    warehouse stores frames. It is *not* a hash of the serialised file.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to fingerprint.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Notes
    -----
    :func:`pandas.util.hash_pandas_object` is a vectorised per-row 64-bit hash, so this
    costs one pass over the data instead of a full serialisation. Row hashes are packed
    native-endian, so digests are comparable within one platform and pandas version --
    which is the intended use (change detection between runs), not cross-platform
    attestation. Frames holding unhashable cell values (lists, dicts) fall back to a
    stringified hash so the call never raises.
    """
    header = json.dumps(
        {"columns": [str(c) for c in df.columns], "dtypes": [str(d) for d in df.dtypes]},
        sort_keys=False,
    ).encode("utf-8")
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False, categorize=False)
    except TypeError:  # unhashable cell contents -- degrade instead of exploding
        row_hashes = pd.util.hash_pandas_object(df.astype(str), index=False, categorize=False)
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"|")
    digest.update(np.ascontiguousarray(row_hashes.to_numpy(dtype="uint64")).tobytes())
    return digest.hexdigest()


def _dtype_map(df: pd.DataFrame) -> dict[str, str]:
    """Return ``{column: dtype string}`` for later restoration."""
    return {str(c): str(d) for c, d in df.dtypes.items()}


def _restore_dtypes(df: pd.DataFrame, mapping: dict[str, str] | None) -> pd.DataFrame:
    """Re-apply a recorded dtype map to a frame read back from storage.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame as returned by the storage engine.
    mapping : dict of str to str or None
        ``{column: dtype string}`` captured at write time. ``None`` is a no-op.

    Returns
    -------
    pandas.DataFrame
        ``df`` with dtypes coerced back where possible. A column whose dtype cannot be
        reconstructed (an exotic extension type, or data that no longer fits) is left
        as-is rather than raising -- reading data back must not be able to fail.
    """
    if not mapping:
        return df
    for col, want in mapping.items():
        if col not in df.columns:
            continue
        have = str(df[col].dtype)
        if have == want:
            continue
        try:
            target = pd.api.types.pandas_dtype(want)
        except TypeError:  # pragma: no cover - unknown dtype string
            continue
        try:
            df[col] = df[col].astype(target)
        except (TypeError, ValueError):  # pragma: no cover - incompatible values
            LOG.debug("could not restore column %s to dtype %s (left as %s)", col, want, have)
    return df


def _as_iso(written_at: str | pd.Timestamp | None) -> str:
    """Normalise a lineage timestamp to an ISO-8601 string.

    Parameters
    ----------
    written_at : str or pandas.Timestamp or None
        Caller-supplied timestamp. ``None`` reads the wall clock.

    Returns
    -------
    str
        ISO-8601 representation.

    Notes
    -----
    This is the one place PRISM touches the wall clock, and it is deliberate: the value
    is *metadata about the run*, never an input to any computation, so determinism is
    unaffected. Pass ``written_at`` explicitly to make lineage byte-reproducible too.
    """
    if written_at is None:
        return pd.Timestamp.now(tz="UTC").isoformat()
    if isinstance(written_at, str):
        return written_at
    return pd.Timestamp(written_at).isoformat()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a frame on its way into storage."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")
    cols = [str(c) for c in df.columns]
    # Counter, not `cols.count(c)` in a comprehension: the latter is O(n_cols**2) and
    # a wide feature matrix makes that measurable for no reason.
    dupes = sorted(c for c, k in Counter(cols).items() if k > 1)
    if dupes:
        raise ValueError(f"cannot store a frame with duplicate column names: {dupes}")
    out = df.reset_index(drop=True)
    out.columns = cols
    return out


# ======================================================================================
# Warehouse
# ======================================================================================
class Warehouse:
    """A bronze/silver/gold medallion warehouse over DuckDB, or parquet as a fallback.

    Parameters
    ----------
    path : str or pathlib.Path, default ``"artifacts/warehouse.duckdb"``
        DuckDB database file. Relative paths resolve against the project root and parent
        directories are created. Under the parquet fallback this file is not created;
        the sibling directory ``<stem>_parquet`` is used instead (see :attr:`parquet_dir`).
    backend : {"auto", "duckdb", "parquet"}, default ``"auto"``
        ``"auto"`` picks DuckDB when importable, else parquet -- except that it stays on
        parquet when there is no duckdb file yet but a populated parquet store already
        exists, so a warehouse written before duckdb was installed does not silently
        read as empty. ``"duckdb"`` raises :class:`ImportError` if duckdb is missing;
        ``"parquet"`` forces the fallback (useful for tests, and exercised by this
        module's smoke test).

    Attributes
    ----------
    backend : str
        ``"duckdb"`` or ``"parquet"`` -- the backend actually in use.
    path : pathlib.Path
        Absolute database path.
    parquet_dir : pathlib.Path
        Absolute directory holding parquet tables under the fallback backend.

    Examples
    --------
    >>> import pandas as pd
    >>> with Warehouse("artifacts/demo.duckdb") as wh:            # doctest: +SKIP
    ...     wh.write("gold", "panel", pd.DataFrame({"a": [1, 2]}))
    ...     wh.read("gold", "panel").shape
    (2, 1)

    Notes
    -----
    Writes are **overwrite** semantics: writing ``("gold", "panel")`` twice replaces the
    table and appends a second lineage row, so the lineage is a full audit history while
    :meth:`tables` reports only what is currently materialised.
    """

    # ---------------------------------------------------------------- construction
    def __init__(self, path: str | Path = DEFAULT_WAREHOUSE_PATH, *, backend: str = "auto") -> None:
        if backend not in {"auto", "duckdb", "parquet"}:
            raise ValueError(f"backend must be 'auto', 'duckdb' or 'parquet', got {backend!r}")

        self._path: Path = _resolve_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._parquet_dir: Path = self._path.parent / f"{self._path.stem}_parquet"

        if backend == "duckdb" and not HAS_DUCKDB:
            require("duckdb")  # raises ImportError with an install hint

        if backend == "auto" and HAS_DUCKDB and not self._path.exists() and _has_parquet_store(self._parquet_dir):
            # A previous run fell back to parquet (duckdb was not installed then) and
            # duckdb has since appeared. Switching to duckdb here would open an empty
            # database and report the existing warehouse as empty -- data silently
            # invisible rather than an error. Stay on the store that actually has data.
            LOG.warning(
                "no duckdb file at %s but a populated parquet store at %s; staying on the "
                "parquet backend. Pass backend='duckdb' to start a fresh duckdb warehouse.",
                self._path,
                self._parquet_dir,
            )
            self._backend: str = "parquet"
        else:
            self._backend = "duckdb" if (backend == "duckdb" or (backend == "auto" and HAS_DUCKDB)) else "parquet"

        self._con: Any | None = None
        self._closed: bool = False
        #: table -> {"layer", "name", "dtypes"} for the currently materialised tables.
        self._registry: dict[str, dict[str, Any]] = {}
        #: append-only lineage rows (mirrors the persisted ``_prism_lineage`` table).
        self._lineage_rows: list[dict[str, Any]] = []

        if self._backend == "duckdb":
            duckdb = require("duckdb")
            self._con = duckdb.connect(str(self._path))
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {_q(_LINEAGE_TABLE)} ("
                '  write_id BIGINT, "table" VARCHAR, layer VARCHAR, name VARCHAR,'
                "  n_rows BIGINT, n_cols BIGINT, written_at VARCHAR, sha256 VARCHAR, dtypes VARCHAR)"
            )
        else:
            self._parquet_dir.mkdir(parents=True, exist_ok=True)

        self._load_lineage()
        LOG.debug("warehouse open (%s) at %s with %d table(s)", self._backend, self._path, len(self._registry))

    # ---------------------------------------------------------------- dunder / props
    def __enter__(self) -> Warehouse:
        """Enter a ``with`` block and return ``self``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Close the warehouse on block exit; never suppresses an exception."""
        self.close()
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "closed" if self._closed else f"{len(self._registry)} tables"
        return f"Warehouse(path={str(self._path)!r}, backend={self._backend!r}, {state})"

    def __contains__(self, key: tuple[str, str] | str) -> bool:  # pragma: no cover - convenience
        """Support ``("gold", "panel") in wh`` and ``"gold_panel" in wh``."""
        if isinstance(key, tuple):
            return self.has(*key)
        self._check_open()
        return self._lookup(str(key).lower()) is not None

    @property
    def backend(self) -> str:
        """``"duckdb"`` or ``"parquet"`` -- the storage engine actually in use."""
        return self._backend

    @property
    def path(self) -> Path:
        """Absolute path of the DuckDB database file."""
        return self._path

    @property
    def parquet_dir(self) -> Path:
        """Absolute directory used by the parquet fallback backend."""
        return self._parquet_dir

    # ---------------------------------------------------------------- naming
    @staticmethod
    def table_name(layer: str, name: str) -> str:
        """Return the physical table name ``"<layer>_<name>"``.

        Parameters
        ----------
        layer : str
            One of :data:`LAYERS`.
        name : str
            Dataset name.

        Returns
        -------
        str
            Physical table identifier, e.g. ``"silver_panel"``.
        """
        return f"{_validate_layer(layer)}_{_validate_name(name)}"

    # ---------------------------------------------------------------- write
    def write(
        self,
        layer: str,
        name: str,
        df: pd.DataFrame,
        *,
        written_at: str | pd.Timestamp | None = None,
    ) -> None:
        """Write (overwriting) a frame into a medallion layer and record lineage.

        Parameters
        ----------
        layer : {"bronze", "silver", "gold"}
            Curation layer.
        name : str
            Dataset name; must be a plain identifier (see :func:`_validate_name`).
        df : pandas.DataFrame
            Frame to store. The index is dropped; column order is preserved.
        written_at : str or pandas.Timestamp, optional
            Lineage timestamp. Defaults to the current UTC time. Pass a fixed value to
            make lineage reproducible.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``layer`` is not a medallion layer, ``name`` is not a valid identifier,
            or ``df`` has duplicate column names.
        TypeError
            If ``df`` is not a DataFrame.
        RuntimeError
            If the warehouse has been closed.
        """
        self._check_open()
        table = self.table_name(layer, name)
        layer_k, name_k = _validate_layer(layer), _validate_name(name)
        frame = _prepare(df)
        dtypes = _dtype_map(frame)

        if self._backend == "duckdb":
            assert self._con is not None
            self._con.register(_STAGE_VIEW, frame)
            try:
                self._con.execute(f"CREATE OR REPLACE TABLE {_q(table)} AS SELECT * FROM {_q(_STAGE_VIEW)}")
            finally:
                self._con.unregister(_STAGE_VIEW)
        else:
            self._parquet_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self._parquet_dir / f"{table}.parquet", index=False, engine="pyarrow")

        self._registry[table] = {"layer": layer_k, "name": name_k, "dtypes": dtypes}
        row = {
            "write_id": len(self._lineage_rows) + 1,
            "table": table,
            "layer": layer_k,
            "name": name_k,
            "n_rows": int(frame.shape[0]),
            "n_cols": int(frame.shape[1]),
            "written_at": _as_iso(written_at),
            "sha256": frame_sha256(frame),
            "dtypes": json.dumps(dtypes),
        }
        self._lineage_rows.append(row)
        self._persist_lineage_row(row)
        LOG.debug("wrote %s (%d x %d)", table, row["n_rows"], row["n_cols"])

    # ---------------------------------------------------------------- read
    def read(self, layer: str, name: str) -> pd.DataFrame:
        """Read a table back, restoring the dtypes it was written with.

        Parameters
        ----------
        layer : {"bronze", "silver", "gold"}
            Curation layer.
        name : str
            Dataset name.

        Returns
        -------
        pandas.DataFrame
            The stored frame with a fresh ``RangeIndex``.

        Raises
        ------
        KeyError
            If the table is absent, with a message listing what *is* available.
        ValueError
            If ``layer`` or ``name`` is malformed.
        """
        self._check_open()
        table = self.table_name(layer, name)
        meta = self._lookup(table)
        if meta is None:
            raise KeyError(self._missing_message(table, layer, name))

        if self._backend == "duckdb":
            assert self._con is not None
            out = self._con.execute(f"SELECT * FROM {_q(table)}").fetchdf()
        else:
            out = pd.read_parquet(self._parquet_dir / f"{table}.parquet", engine="pyarrow")

        out = out.reset_index(drop=True)
        return _restore_dtypes(out, meta.get("dtypes"))

    # ---------------------------------------------------------------- sql
    def sql(self, query: str) -> pd.DataFrame:
        """Run a SQL query against the warehouse.

        Parameters
        ----------
        query : str
            SQL text. Tables are addressed by their physical name, ``"<layer>_<name>"``.

        Returns
        -------
        pandas.DataFrame
            Query result.

        Raises
        ------
        NotImplementedError
            Under the parquet fallback, for anything but ``SELECT * FROM <table>``.
        KeyError
            Under the parquet fallback, if the named table does not exist.
        RuntimeError
            If the warehouse has been closed.

        Notes
        -----
        **DuckDB backend** -- the query is executed verbatim by the engine; joins,
        aggregates, window functions and CTEs all work.

        **Parquet fallback** -- there is no SQL engine, so arbitrary SQL cannot be
        honoured and this method raises :class:`NotImplementedError` rather than silently
        returning something wrong. The one shape that *is* supported, because it is by far
        the most common and needs no engine, is a bare ``SELECT * FROM <table>`` (matched
        with a regex, optional trailing semicolon, optional quoting of the identifier,
        case-insensitive); it is dispatched to :meth:`read`. Code that must work on both
        backends should use :meth:`read` and do its filtering in pandas.
        """
        self._check_open()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty SQL string")
        match = _SELECT_STAR_RE.match(query)

        # Physical table names are always lower-cased by `_validate_name`, and SQL
        # identifiers are case-insensitive. Folding the captured identifier keeps
        # `SELECT * FROM GOLD_PANEL` behaving identically on both backends -- before,
        # duckdb served it with unrestored (object) dtypes while parquet raised KeyError.
        hit = match.group(1).lower() if match is not None else None

        if self._backend == "duckdb":
            assert self._con is not None
            out = self._con.execute(query).fetchdf().reset_index(drop=True)
            # Keep `SELECT * FROM <table>` dtype-identical to read() on both backends.
            if hit is not None:
                meta = self._lookup(hit)
                if meta is not None:
                    out = _restore_dtypes(out, meta.get("dtypes"))
            return out

        if match is None:
            raise NotImplementedError(
                "duckdb is not installed, so this Warehouse is running on the parquet "
                "fallback, which has no SQL engine. Only 'SELECT * FROM <table>' is "
                "supported here; use Warehouse.read(layer, name) and filter in pandas, "
                "or install duckdb (pip install duckdb) for full SQL.\n"
                f"  rejected query: {query.strip()[:200]}"
            )
        assert hit is not None
        meta = self._lookup(hit)
        if meta is None:
            raise KeyError(self._missing_message(hit, None, None))
        return self.read(meta["layer"], meta["name"])

    # ---------------------------------------------------------------- catalogue
    def tables(self) -> pd.DataFrame:
        """List the tables currently materialised in the warehouse.

        Returns
        -------
        pandas.DataFrame
            Columns ``table``, ``layer``, ``name``, ``n_rows``, ``n_cols``, sorted by
            medallion layer (bronze, silver, gold) then name.

        Notes
        -----
        Both the table list and the row/column counts come from the live catalogue, not
        from lineage, so a table written by another process (or by raw
        :meth:`sql` DDL) is reported, and a table deleted underneath us is not.
        """
        self._check_open()
        self._refresh_registry()
        records: list[dict[str, Any]] = []
        for table, meta in self._registry.items():
            n_rows, n_cols = self._shape_of(table)
            records.append(
                {
                    "table": table,
                    "layer": meta["layer"],
                    "name": meta["name"],
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                }
            )
        out = pd.DataFrame(records, columns=list(_TABLE_COLUMNS))
        if out.empty:
            return out.astype({"n_rows": "int64", "n_cols": "int64"})
        order = pd.Categorical(out["layer"], categories=list(LAYERS), ordered=True)
        out = out.assign(_o=order).sort_values(["_o", "name"]).drop(columns="_o").reset_index(drop=True)
        return out.astype({"n_rows": "int64", "n_cols": "int64"})

    def lineage(self, *, full: bool = False, latest_only: bool = False) -> pd.DataFrame:
        """Return the write-audit history, persisted across sessions.

        Parameters
        ----------
        full : bool, default False
            Also return the internal columns ``write_id``, ``name`` and ``dtypes``
            (the JSON dtype map used for round-trip restoration).
        latest_only : bool, default False
            Keep only the most recent write per table instead of the full history.

        Returns
        -------
        pandas.DataFrame
            Columns ``table``, ``layer``, ``n_rows``, ``n_cols``, ``written_at``,
            ``sha256`` (plus the internal columns when ``full=True``), oldest first.
            The history survives :meth:`close` because it is stored as its own table
            (``_prism_lineage`` in DuckDB, ``_prism_lineage.parquet`` in the fallback).
        """
        if self._lineage_rows:
            out = pd.DataFrame(self._lineage_rows, columns=list(_LINEAGE_STORE_COLUMNS))
        else:
            out = _EMPTY_LINEAGE.copy()
        out = out.astype({"write_id": "int64", "n_rows": "int64", "n_cols": "int64"})
        if latest_only:
            out = out.drop_duplicates(subset="table", keep="last")
        cols = list(_LINEAGE_STORE_COLUMNS) if full else list(_LINEAGE_COLUMNS)
        return out[cols].reset_index(drop=True)

    def has(self, layer: str, name: str) -> bool:
        """Return whether ``<layer>_<name>`` is currently materialised.

        Parameters
        ----------
        layer : {"bronze", "silver", "gold"}
            Curation layer.
        name : str
            Dataset name.

        Returns
        -------
        bool

        Raises
        ------
        RuntimeError
            If the warehouse has been closed. (Returning ``False`` from a dead handle
            would be indistinguishable from "the table is not there".)
        """
        self._check_open()
        return self._lookup(self.table_name(layer, name)) is not None

    def drop(self, layer: str, name: str, *, missing_ok: bool = False) -> None:
        """Delete a table. Lineage rows are kept -- they are the audit trail.

        Parameters
        ----------
        layer : {"bronze", "silver", "gold"}
            Curation layer.
        name : str
            Dataset name.
        missing_ok : bool, default False
            Return quietly instead of raising when the table is absent.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the table is absent and ``missing_ok`` is False.
        """
        self._check_open()
        table = self.table_name(layer, name)
        if self._lookup(table) is None:
            if missing_ok:
                return
            raise KeyError(self._missing_message(table, layer, name))
        if self._backend == "duckdb":
            assert self._con is not None
            self._con.execute(f"DROP TABLE IF EXISTS {_q(table)}")
        else:
            (self._parquet_dir / f"{table}.parquet").unlink(missing_ok=True)
        self._registry.pop(table, None)

    def close(self) -> None:
        """Flush and release the backend. Idempotent, and safe to call twice.

        Closing the DuckDB connection is what releases the Windows file lock, so the
        database file can afterwards be moved or deleted.

        Returns
        -------
        None
        """
        if self._closed:
            return
        if self._con is not None:
            try:
                self._con.close()
            except Exception:  # pragma: no cover - already-dead connection
                LOG.debug("duckdb connection close raised; ignoring")
            self._con = None
        self._closed = True

    # ---------------------------------------------------------------- internals
    def _check_open(self) -> None:
        """Raise if the warehouse has been closed."""
        if self._closed:
            raise RuntimeError(f"Warehouse at {self._path} is closed; open a new one to continue")

    def _exists(self, table: str) -> bool:
        """Return whether the physical table/file is present right now."""
        if self._backend == "duckdb":
            if self._con is None:
                return False
            hit = self._con.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name = ?",
                [table],
            ).fetchone()
            return bool(hit and hit[0])
        return (self._parquet_dir / f"{table}.parquet").exists()

    def _discover(self) -> dict[str, tuple[str, str]]:
        """Enumerate the medallion tables physically present *right now*.

        Returns
        -------
        dict of str to tuple of (str, str)
            ``{physical_table: (layer, name)}``. Internal ``_prism*`` objects and
            anything not shaped like ``<layer>_<name>`` are ignored.
        """
        found: dict[str, tuple[str, str]] = {}
        if self._backend == "duckdb":
            if self._con is None:
                return found
            names = [
                str(r[0])
                for r in self._con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
            ]
        else:
            if not self._parquet_dir.is_dir():
                return found
            names = [p.stem for p in self._parquet_dir.glob("*.parquet")]
        for table in names:
            key = table.lower()
            if key.startswith(_RESERVED_PREFIX):
                continue
            layer, _, name = key.partition("_")
            if layer in LAYERS and _NAME_RE.match(name):
                found[key] = (layer, name)
        return found

    def _refresh_registry(self) -> None:
        """Reconcile the in-memory registry with the live catalogue.

        Tables present on disk but unknown to lineage are adopted (with an empty dtype
        map, so :meth:`read` returns them in the storage engine's native dtypes), and
        tables whose storage has disappeared are forgotten. Without this the warehouse
        is blind to any table it did not itself write in this or a previous session --
        including one produced by another tool or by raw ``CREATE TABLE`` through
        :meth:`sql`.
        """
        live = self._discover()
        for table, (layer, name) in live.items():
            if table not in self._registry:
                self._registry[table] = {"layer": layer, "name": name, "dtypes": {}}
        for table in [t for t in self._registry if t not in live]:
            self._registry.pop(table, None)

    def _lookup(self, table: str) -> dict[str, Any] | None:
        """Return live metadata for ``table``, re-scanning the catalogue on a miss.

        Returns
        -------
        dict or None
            ``None`` when the table is genuinely not materialised. A non-``None``
            result is guaranteed to refer to storage that exists.
        """
        meta = self._registry.get(table)
        if meta is not None and self._exists(table):
            return meta
        self._refresh_registry()
        return self._registry.get(table)

    def _shape_of(self, table: str) -> tuple[int, int]:
        """Return ``(n_rows, n_cols)`` of a materialised table without loading it."""
        if self._backend == "duckdb":
            assert self._con is not None
            n_rows = int(self._con.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()[0])
            n_cols = int(
                self._con.execute(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_schema = 'main' AND table_name = ?",
                    [table],
                ).fetchone()[0]
            )
            return n_rows, n_cols
        import pyarrow.parquet as pq  # hard dependency; imported lazily to keep import cheap

        meta = pq.ParquetFile(self._parquet_dir / f"{table}.parquet").metadata
        return int(meta.num_rows), int(meta.num_columns)

    def _missing_message(self, table: str, layer: str | None, name: str | None) -> str:
        """Build a helpful KeyError message that lists the available tables."""
        self._refresh_registry()
        available = sorted(self._registry)
        wanted = f"{layer}/{name}" if layer is not None else table
        hint = ", ".join(available) if available else "(warehouse is empty)"
        return (
            f"table {table!r} (layer/name {wanted}) is not in the warehouse at {self._path}. "
            f"Available: {hint}. Write it first with Warehouse.write(layer, name, df)."
        )

    def _lineage_store_path(self) -> Path:
        """Parquet path of the persisted lineage table (fallback backend)."""
        return self._parquet_dir / f"{_LINEAGE_TABLE}.parquet"

    def _persist_lineage_row(self, row: dict[str, Any]) -> None:
        """Append one lineage row to durable storage."""
        if self._backend == "duckdb":
            assert self._con is not None
            self._con.execute(
                f"INSERT INTO {_q(_LINEAGE_TABLE)} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    int(row["write_id"]),
                    row["table"],
                    row["layer"],
                    row["name"],
                    int(row["n_rows"]),
                    int(row["n_cols"]),
                    row["written_at"],
                    row["sha256"],
                    row["dtypes"],
                ],
            )
        else:
            frame = pd.DataFrame(self._lineage_rows, columns=list(_LINEAGE_STORE_COLUMNS))
            frame.to_parquet(self._lineage_store_path(), index=False, engine="pyarrow")

    def _load_lineage(self) -> None:
        """Restore lineage and the table registry from a previous session.

        The registry is seeded from the live catalogue (:meth:`_refresh_registry`) and
        lineage only supplies the recorded dtype maps on top. Doing it in that order --
        rather than trusting lineage alone -- is what lets the warehouse read a table
        that exists on disk but was never written through this class.
        """
        stored: pd.DataFrame | None = None
        if self._backend == "duckdb":
            assert self._con is not None
            stored = self._con.execute(
                f"SELECT * FROM {_q(_LINEAGE_TABLE)} ORDER BY write_id"
            ).fetchdf()
        else:
            store = self._lineage_store_path()
            if store.exists():
                stored = pd.read_parquet(store, engine="pyarrow")

        if stored is not None and not stored.empty:
            stored = stored[list(_LINEAGE_STORE_COLUMNS)].sort_values("write_id")
            self._lineage_rows = stored.to_dict("records")

        self._refresh_registry()
        # Overlay the recorded dtype maps onto whatever is actually materialised.
        for row in self._lineage_rows:
            table = str(row["table"])
            if table not in self._registry:
                continue  # lineage remembers it; storage no longer has it
            try:
                dtypes = json.loads(row["dtypes"]) if row.get("dtypes") else {}
            except (TypeError, ValueError):  # pragma: no cover - corrupted metadata
                dtypes = {}
            self._registry[table] = {
                "layer": str(row["layer"]),
                "name": str(row["name"]),
                "dtypes": dtypes,
            }


# ======================================================================================
# convenience
# ======================================================================================
def build_medallion(
    bronze_events: pd.DataFrame,
    bronze_customers: pd.DataFrame,
    silver_panel: pd.DataFrame,
    gold_panel: pd.DataFrame,
    path: str | Path = DEFAULT_WAREHOUSE_PATH,
    *,
    backend: str = "auto",
    written_at: str | pd.Timestamp | None = None,
) -> Warehouse:
    """Populate the standard PRISM medallion in one call and return the open warehouse.

    Writes four tables: ``bronze_events``, ``bronze_customers``, ``silver_panel`` and
    ``gold_panel`` -- the canonical layout the rest of the pipeline reads from.

    Parameters
    ----------
    bronze_events : pandas.DataFrame
        Raw event log as landed (source for the point-in-time feature store).
    bronze_customers : pandas.DataFrame
        Raw customer dimension, one row per customer.
    silver_panel : pandas.DataFrame
        Cleaned, conformed panel (output of ``prism.data.messy.clean_panel``).
    gold_panel : pandas.DataFrame
        Contract-validated analysis panel (``PANEL_CONTRACT``).
    path : str or pathlib.Path, default ``"artifacts/warehouse.duckdb"``
        Warehouse location; parent directories are created.
    backend : {"auto", "duckdb", "parquet"}, default ``"auto"``
        Forwarded to :class:`Warehouse`.
    written_at : str or pandas.Timestamp, optional
        Fixed lineage timestamp applied to all four writes.

    Returns
    -------
    Warehouse
        The **open** warehouse. The caller owns it: call :meth:`Warehouse.close` or use
        it as a context manager.

    Examples
    --------
    >>> wh = build_medallion(events, customers, silver, gold)   # doctest: +SKIP
    >>> wh.tables()                                             # doctest: +SKIP
    >>> wh.close()                                              # doctest: +SKIP
    """
    wh = Warehouse(path, backend=backend)
    try:
        wh.write("bronze", "events", bronze_events, written_at=written_at)
        wh.write("bronze", "customers", bronze_customers, written_at=written_at)
        wh.write("silver", "panel", silver_panel, written_at=written_at)
        wh.write("gold", "panel", gold_panel, written_at=written_at)
    except Exception:
        wh.close()
        raise
    LOG.info("medallion built at %s (%s backend, %d tables)", wh.path, wh.backend, len(wh.tables()))
    return wh


# ======================================================================================
# smoke test
# ======================================================================================
def _demo_frame(n: int, seed: int) -> pd.DataFrame:
    """Build a small frame exercising string / int64 / float64 / datetime64 dtypes."""
    from prism.utils.seeds import as_rng

    rng = as_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": pd.array([f"C{i:05d}" for i in range(n)], dtype="string"),
            "period": np.arange(n, dtype="int64") % 4,
            "value_h": rng.normal(100.0, 10.0, size=n).astype("float64"),
            "as_of_date": pd.to_datetime("2022-01-01") + pd.to_timedelta(np.arange(n) * 30, unit="D"),
            "plan_tier": pd.array(
                [["basic", "plus", "pro", "enterprise"][i % 4] for i in range(n)], dtype="string"
            ),
        }
    )


def _purge(path: Path) -> None:
    """Remove a warehouse file, its DuckDB write-ahead log and any parquet sidecar."""
    for p in (path, path.with_name(path.name + ".wal"), path.with_name(path.name + ".tmp")):
        try:
            p.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - locked by another process
            pass
    sidecar = path.parent / f"{path.stem}_parquet"
    if sidecar.exists():
        shutil.rmtree(sidecar, ignore_errors=True)


def _run_backend_checks(backend: str, path: Path) -> dict[str, Any]:
    """Exercise one backend end-to-end and return a compact result record."""
    _purge(path)
    stamp = "2024-01-01T00:00:00+00:00"
    bronze = _demo_frame(200, seed=1)
    silver = _demo_frame(150, seed=2)
    gold = _demo_frame(100, seed=3)
    gold_pristine = gold.copy(deep=True)

    # 0. determinism / sensitivity of the content hash, before anything touches disk.
    assert frame_sha256(gold) == frame_sha256(_demo_frame(100, seed=3)), "sha256 is not deterministic"
    assert frame_sha256(gold) != frame_sha256(gold.iloc[::-1].reset_index(drop=True)), "sha256 ignores row order"
    assert frame_sha256(gold) != frame_sha256(gold.head(99)), "sha256 ignores row count"

    with Warehouse(path, backend=backend) as wh:
        assert wh.backend == backend, (wh.backend, backend)
        wh.write("bronze", "events", bronze, written_at=stamp)
        wh.write("silver", "panel", silver, written_at=stamp)
        wh.write("gold", "panel", gold, written_at=stamp)

        # 1. dtype + value round-trip across all three layers.
        for layer, original in (("bronze", bronze), ("silver", silver), ("gold", gold)):
            name = "events" if layer == "bronze" else "panel"
            back = wh.read(layer, name)
            pd.testing.assert_frame_equal(back, original, check_dtype=True, check_exact=True)
        assert str(wh.read("gold", "panel")["customer_id"].dtype) == "string"
        assert str(wh.read("gold", "panel")["as_of_date"].dtype) == "datetime64[ns]"

        # 2. overwrite semantics: second write replaces rows, adds a lineage row.
        wh.write("gold", "panel", gold.head(10), written_at=stamp)
        assert len(wh.read("gold", "panel")) == 10
        assert int((wh.lineage()["table"] == "gold_panel").sum()) == 2
        wh.write("gold", "panel", gold, written_at=stamp)

        # 3. SELECT * works on both backends and keeps dtypes.
        via_sql = wh.sql("SELECT * FROM gold_panel")
        pd.testing.assert_frame_equal(via_sql, gold, check_dtype=True, check_exact=True)

        # 4. arbitrary SQL: engine under duckdb, clear NotImplementedError under parquet.
        agg_query = "SELECT period, count(*) AS n FROM gold_panel GROUP BY period ORDER BY period"
        if backend == "duckdb":
            agg = wh.sql(agg_query)
            assert int(agg["n"].sum()) == len(gold), agg
            sql_note = f"aggregate ok ({len(agg)} groups)"
        else:
            try:
                wh.sql(agg_query)
            except NotImplementedError as exc:
                assert "parquet fallback" in str(exc)
                sql_note = "NotImplementedError raised as documented"
            else:  # pragma: no cover - would be a bug
                raise AssertionError("parquet fallback accepted arbitrary SQL")

        # 5. error contracts.
        try:
            wh.write("platinum", "panel", gold)
        except ValueError as exc:
            assert "bronze" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("bad layer was accepted")
        try:
            wh.read("gold", "nonexistent")
        except KeyError as exc:
            assert "Available" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("missing table did not raise KeyError")

        # 6. SQL identifiers are case-insensitive, identically on both backends.
        #    (Regression: duckdb used to serve 'GOLD_PANEL' with unrestored object
        #    dtypes while the parquet fallback raised KeyError for the same query.)
        for spelling in ("SELECT * FROM gold_panel", "SELECT * FROM GOLD_PANEL", "select * from Gold_Panel;"):
            pd.testing.assert_frame_equal(wh.sql(spelling), gold, check_dtype=True, check_exact=True)

        # 7. tables the warehouse did not write itself are still discoverable.
        #    (Regression: the registry was rebuilt from lineage alone, so a table
        #    produced by another tool was invisible to tables()/has()/read() even
        #    though duckdb's sql() would happily read it.)
        if backend == "duckdb":
            wh.sql("CREATE TABLE bronze_foreign AS SELECT 1 AS a, 2 AS b")
        else:
            pd.DataFrame({"a": [1], "b": [2]}).to_parquet(
                wh.parquet_dir / "bronze_foreign.parquet", index=False, engine="pyarrow"
            )
        assert wh.has("bronze", "foreign"), "externally created table is invisible to has()"
        assert "bronze_foreign" in set(wh.tables()["table"]), "externally created table missing from tables()"
        assert wh.read("bronze", "foreign").shape == (1, 2)
        wh.drop("bronze", "foreign")
        assert not wh.has("bronze", "foreign")

        # 8. lineage sha256 tracks content: equal writes agree, the short write differs.
        gold_shas = wh.lineage().loc[lambda d: d["table"] == "gold_panel", "sha256"].tolist()
        assert len(gold_shas) == 3 and gold_shas[0] == gold_shas[2] != gold_shas[1], gold_shas

        # 9. writing must not mutate the caller's frame.
        pd.testing.assert_frame_equal(gold, gold_pristine, check_dtype=True, check_exact=True)
        assert gold.index.equals(gold_pristine.index), "write() reset the caller's index"

        tables_before = wh.tables()
        lineage_before = wh.lineage()
        sha_before = lineage_before.iloc[-1]["sha256"]

    # 6. persistence across sessions (lineage is its own table).
    with Warehouse(path, backend=backend) as wh2:
        assert wh2.has("gold", "panel")
        pd.testing.assert_frame_equal(wh2.read("gold", "panel"), gold, check_dtype=True)
        assert len(wh2.lineage()) == len(lineage_before)
        assert wh2.lineage().iloc[-1]["sha256"] == sha_before
        assert frame_sha256(gold) == sha_before
        reopened_tables = len(wh2.tables())
        wh2.drop("silver", "panel")
        assert not wh2.has("silver", "panel")
        assert len(wh2.lineage()) == len(lineage_before)  # audit history is retained

    # 10. closed warehouse rejects further use -- including the catalogue queries.
    #     has() must raise rather than answer False, which would be indistinguishable
    #     from "the table is not there".
    wh3 = Warehouse(path, backend=backend)
    wh3.close()
    wh3.close()  # idempotent
    for probe in (
        lambda: wh3.read("gold", "panel"),
        lambda: wh3.tables(),
        lambda: wh3.has("gold", "panel"),
        lambda: wh3.sql("SELECT * FROM gold_panel"),
    ):
        try:
            probe()
        except RuntimeError:
            continue
        raise AssertionError("closed warehouse still served requests")

    _purge(path)
    residue = [p.name for p in path.parent.glob(f"{path.stem}*")]
    return {
        "backend": backend,
        "tables": len(tables_before),
        "lineage_rows": len(lineage_before),
        "reopened_tables": reopened_tables,
        "sql": sql_note,
        "residue": residue,
        "tables_frame": tables_before,
        "lineage_frame": lineage_before,
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    t0 = time.perf_counter()
    smoke_path = _resolve_path("artifacts/_wh_smoke.duckdb")
    smoke_path.parent.mkdir(parents=True, exist_ok=True)

    backends = ["duckdb", "parquet"] if HAS_DUCKDB else ["parquet"]
    results: list[dict[str, Any]] = []
    for be in backends:
        results.append(_run_backend_checks(be, smoke_path))

    shown = results[0]
    print("=" * 78)
    print(f"PRISM warehouse smoke test  (duckdb available: {HAS_DUCKDB})")
    print("=" * 78)
    print(f"\n[{shown['backend']}] tables():")
    print(shown["tables_frame"].to_string(index=False))
    print(f"\n[{shown['backend']}] lineage():")
    print(shown["lineage_frame"].assign(sha256=lambda d: d["sha256"].str.slice(0, 12) + "...").to_string(index=False))

    # build_medallion end-to-end, on a separate temp path.
    med_path = _resolve_path("artifacts/_wh_smoke_medallion.duckdb")
    _purge(med_path)
    with build_medallion(
        _demo_frame(60, 11), _demo_frame(20, 12), _demo_frame(40, 13), _demo_frame(30, 14),
        path=med_path, written_at="2024-01-01T00:00:00+00:00",
    ) as med:
        med_tables = med.tables()
        med_backend = med.backend
        # tables() sorts by medallion layer then name, so the order is deterministic.
        assert list(med_tables["table"]) == [
            "bronze_customers",
            "bronze_events",
            "silver_panel",
            "gold_panel",
        ], list(med_tables["table"])
        assert list(med_tables["layer"]) == ["bronze", "bronze", "silver", "gold"]
        assert med_tables["n_rows"].tolist() == [20, 60, 40, 30], med_tables["n_rows"].tolist()
        assert len(med.read("gold", "panel")) == 30
        assert len(med.lineage()) == 4 and med.lineage()["written_at"].nunique() == 1
    _purge(med_path)

    print(f"\n[build_medallion / {med_backend}] tables():")
    print(med_tables.to_string(index=False))

    # backend='auto' must not orphan a warehouse written before duckdb was installed.
    # (Regression: auto opened an empty duckdb file next to a populated parquet store
    # and reported the warehouse as empty -- data silently invisible, no error.)
    auto_path = _resolve_path("artifacts/_wh_smoke_auto.duckdb")
    _purge(auto_path)
    with Warehouse(auto_path, backend="parquet") as pq_wh:
        pq_wh.write("gold", "panel", _demo_frame(7, 21), written_at="2024-01-01T00:00:00+00:00")
    with Warehouse(auto_path, backend="auto") as auto_wh:
        assert auto_wh.backend == "parquet", auto_wh.backend
        assert len(auto_wh.read("gold", "panel")) == 7, "auto backend orphaned the parquet store"
    _purge(auto_path)
    print("\nauto-backend reopen of a parquet store: kept parquet, 7 rows intact")

    print("\n" + "-" * 78)
    print(f"{'backend':10s} {'tables':>7s} {'lineage':>8s} {'reopened':>9s}  {'residue':>8s}  sql behaviour")
    for r in results:
        print(
            f"{r['backend']:10s} {r['tables']:>7d} {r['lineage_rows']:>8d} "
            f"{r['reopened_tables']:>9d}  {str(len(r['residue'])):>8s}  {r['sql']}"
        )
    leftovers = sorted(p.name for p in smoke_path.parent.glob("_wh_smoke*"))
    assert not leftovers, f"smoke test left residue: {leftovers}"
    print("-" * 78)
    print(f"no residue left behind | elapsed {time.perf_counter() - t0:.2f}s")
    print("warehouse.py OK")
