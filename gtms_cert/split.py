"""Exact minimax split of a giant tour."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .geo import DistanceOracle


def _seg_cost(a: int, b: int, tour: Sequence[str], depot: str, d: DistanceOracle, prefix: List[float]) -> float:
    start = tour[a]
    end = tour[b]
    return d[depot][start] + (prefix[b] - prefix[a]) + d[end][depot]


def _feasible(limit: float, tour: Sequence[str], depot: str, d: DistanceOracle, prefix: List[float], k: int) -> bool:
    segments = 0
    a = 0
    n = len(tour)
    while a < n:
        b = a
        while b + 1 < n and _seg_cost(a, b + 1, tour, depot, d, prefix) <= limit:
            b += 1
        segments += 1
        a = b + 1
        if segments > k:
            return False
    return segments <= k


def minimax_split(tour: Sequence[str], depot: str, d: DistanceOracle, k: int) -> tuple[list[tuple[int, int]], float]:
    n = len(tour)
    if n == 0:
        raise ValueError("Tour cannot be empty")
    prefix = [0.0] * (n)
    for j in range(1, n):
        prefix[j] = prefix[j - 1] + d[tour[j - 1]][tour[j]]
    low = max(2 * min(d[depot][i], d[i][depot]) for i in tour)
    high = sum(d[tour[j - 1]][tour[j]] for j in range(1, n)) + min(
        d[depot][tour[0]] + d[tour[-1]][depot],
        d[depot][tour[-1]] + d[tour[0]][depot],
    )
    while high - low > 1e-6:
        mid = (low + high) / 2
        if _feasible(mid, tour, depot, d, prefix, k):
            high = mid
        else:
            low = mid
    z_star = high
    parts: list[tuple[int, int]] = []
    a = 0
    while a < n and len(parts) < k - 1:
        b = a
        remaining_segments = (k - 1) - len(parts)
        while b + 1 < n - remaining_segments and _seg_cost(a, b + 1, tour, depot, d, prefix) <= z_star:
            b += 1
        parts.append((a, b))
        a = b + 1
    parts.append((a, n - 1))
    return parts, z_star
