"""Construction of a giant tour using a simple heuristic."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence

from .geo import DistanceOracle


def build_giant_tour(clients: Sequence[str], oracle: DistanceOracle, seed: int = 42, max_2opt_iter: int = 5) -> List[str]:
    """Construct a tour visiting all clients once."""
    if not clients:
        return []
    rng = random.Random(seed)
    unvisited = set(clients)
    start = rng.choice(list(unvisited))
    tour = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        candidate = _nearest_neighbor(current, unvisited, oracle)
        tour.append(candidate)
        unvisited.remove(candidate)
        current = candidate
    _two_opt_improvement(tour, oracle, iterations=max_2opt_iter)
    return tour


def _nearest_neighbor(current: str, candidates: set[str], oracle: DistanceOracle) -> str:
    neighs = [n for n, _ in oracle.neighbors(current) if n in candidates]
    if neighs:
        return min(neighs, key=lambda n: oracle.get(current, n))
    # Fallback to full search
    return min(candidates, key=lambda n: oracle.get(current, n))


def _two_opt_improvement(tour: List[str], oracle: DistanceOracle, iterations: int = 5) -> None:
    n = len(tour)
    if n < 4:
        return
    for _ in range(iterations):
        improved = False
        index_map = {node: idx for idx, node in enumerate(tour)}
        for i in range(n - 3):
            a, b = tour[i], tour[i + 1]
            for neighbor, _ in oracle.neighbors(a):
                j = index_map.get(neighbor)
                if j is None or j <= i + 1 or j >= n - 1:
                    continue
                c = tour[j]
                d = tour[j + 1]
                delta = (
                    oracle.get(a, c)
                    + oracle.get(b, d)
                    - oracle.get(a, b)
                    - oracle.get(c, d)
                )
                if delta < -1e-6:
                    segment = tour[i + 1 : j + 1]
                    segment.reverse()
                    tour[i + 1 : j + 1] = segment
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
