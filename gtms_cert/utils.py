"""Utility helpers for the GTMS-Cert solver."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple


LOGGER = logging.getLogger("gtms_cert")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the solver logger once."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for determinism."""
    random.seed(seed)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.elapsed = time.perf_counter() - (self.start_time or time.perf_counter())


@dataclass
class Route:
    """Simple representation of a depot-to-depot route."""

    vehicle: int
    sequence: List[str]
    time_min: float


def log_progress(ub: float, lb: float, gap: float) -> None:
    """Log the current optimisation state."""
    LOGGER.info("UB=%.3f LB=%.3f gap=%.3f%%", ub, lb, 100 * gap)


def pairwise(sequence: Sequence[str]) -> Iterator[Tuple[str, str]]:
    """Yield consecutive pairs from a sequence."""
    for i in range(len(sequence) - 1):
        yield sequence[i], sequence[i + 1]
