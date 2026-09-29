"""Consistent logging for pipeline steps, with `rich` formatting when available."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prism.utils.optional import HAS_RICH

__all__ = ["get_logger", "timed", "set_level"]

_CONFIGURED = False
_LEVEL = logging.INFO


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("prism")
    root.setLevel(_LEVEL)
    root.propagate = False
    if root.handlers:
        _CONFIGURED = True
        return
    if HAS_RICH:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        except Exception:  # pragma: no cover
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``prism`` root.

    Parameters
    ----------
    name : str
        Logger name; ``prism.`` is prefixed if absent.

    Returns
    -------
    logging.Logger
    """
    _configure()
    if not name.startswith("prism"):
        name = f"prism.{name}"
    return logging.getLogger(name)


def set_level(level: int | str) -> None:
    """Set the level of the whole ``prism`` logger tree."""
    global _LEVEL
    _LEVEL = logging.getLevelName(level) if isinstance(level, str) else level
    _configure()
    logging.getLogger("prism").setLevel(_LEVEL)


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log the wall-clock duration of a block.

    Parameters
    ----------
    logger : logging.Logger
        Destination logger.
    label : str
        Human-readable step name.
    """
    logger.info("%s ...", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s done in %.2fs", label, time.perf_counter() - t0)


if __name__ == "__main__":  # pragma: no cover - smoke test
    log = get_logger("demo")
    with timed(log, "sleepy step"):
        sum(range(100000))
    log.info("logging.py OK")
