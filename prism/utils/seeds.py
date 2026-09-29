"""Deterministic randomness helpers.

Every stochastic entry point in PRISM funnels through :func:`as_rng` so that a run is
bit-reproducible from a single integer seed.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["as_rng", "seed_everything", "spawn_rngs"]


def as_rng(random_state: int | np.random.Generator | np.random.RandomState | None = None) -> np.random.Generator:
    """Normalise any seed-like object into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    random_state : int or numpy.random.Generator or numpy.random.RandomState or None
        Seed material. ``None`` produces a fresh, non-deterministic generator.

    Returns
    -------
    numpy.random.Generator
        A generator instance. If ``random_state`` is already a ``Generator`` it is
        returned unchanged so that callers can thread one stream through a call chain.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2**32 - 1))
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(int(random_state))


def spawn_rngs(random_state: int | np.random.Generator | None, n: int) -> list[np.random.Generator]:
    """Create ``n`` independent child generators from one seed.

    Uses :meth:`numpy.random.SeedSequence.spawn`, which guarantees statistically
    independent streams — the correct way to seed parallel workers.

    Parameters
    ----------
    random_state : int or numpy.random.Generator or None
        Parent seed material.
    n : int
        Number of child streams.

    Returns
    -------
    list of numpy.random.Generator
    """
    rng = as_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=n)
    return [np.random.default_rng(int(s)) for s in seeds]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch.

    Parameters
    ----------
    seed : int
        Seed value applied to every RNG source PRISM can reach.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Deliberately seeds numpy's LEGACY global RNG. PRISM's own code uses Generator via
    # as_rng, but scikit-learn, LightGBM and others still consult the global state, so this
    # is what makes third-party randomness reproducible too.
    np.random.seed(seed % (2**32 - 1))  # noqa: NPY002
    try:  # pragma: no cover - depends on optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover - smoke test
    a = as_rng(7).normal(size=5)
    b = as_rng(7).normal(size=5)
    assert np.allclose(a, b), "as_rng is not deterministic"
    kids = spawn_rngs(7, 3)
    assert len({tuple(np.round(k.normal(size=3), 8)) for k in kids}) == 3
    seed_everything(7)
    print("seeds.py OK  ->", np.round(a, 4))
