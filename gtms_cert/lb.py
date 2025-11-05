"""Lower bounds via Held-Karp 1-tree relaxation."""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Tuple

from .geo import DistanceOracle


def held_karp_1tree_lb(oracle: DistanceOracle, iterations: int = 200, step0: float = 1.0) -> float:
    """Compute a Held-Karp lower bound using a subgradient scheme."""
    nodes = oracle.nodes()
    depot = oracle.depot_id
    pi = {node: 0.0 for node in nodes}
    best_lb = 0.0
    for t in range(iterations):
        def cprime(i: str, j: str) -> float:
            return oracle.get(i, j) + pi[i] + pi[j]

        cost, degrees = _min_1tree_cost_and_degrees(oracle, cprime)
        lb = cost - 2.0 * sum(pi.values())
        if lb > best_lb:
            best_lb = lb
        g = {node: (degrees.get(node, 0) - 2) if node != depot else 0.0 for node in nodes}
        norm = sum(val * val for val in g.values()) ** 0.5
        if norm < 1e-9:
            break
        step = step0 / (1 + t) ** 0.5
        for node in nodes:
            pi[node] += step * g[node]
    return best_lb


def _min_1tree_cost_and_degrees(
    oracle: DistanceOracle, cprime: callable
) -> Tuple[float, Dict[str, int]]:
    depot = oracle.depot_id
    nodes = [node for node in oracle.nodes() if node != depot]
    if not nodes:
        return 0.0, {depot: 0}
    cost = 0.0
    degrees: Dict[str, int] = defaultdict(int)
    # Prim's algorithm on candidate graph
    visited = set()
    start = nodes[0]
    visited.add(start)
    heap: List[tuple[float, str, str]] = []
    for neigh, _ in oracle.neighbors(start):
        if neigh == depot:
            continue
        heap.append((cprime(start, neigh), start, neigh))
    heapq.heapify(heap)
    while heap and len(visited) < len(nodes):
        w, u, v = heapq.heappop(heap)
        if v in visited or v == depot:
            continue
        visited.add(v)
        cost += w
        degrees[u] += 1
        degrees[v] += 1
        for neigh, _ in oracle.neighbors(v):
            if neigh == depot or neigh in visited:
                continue
            heapq.heappush(heap, (cprime(v, neigh), v, neigh))
    if len(visited) < len(nodes):
        # Graph not fully connected via candidates; fall back to dense Prim
        remaining = set(nodes) - visited
        for node in remaining:
            visited.add(node)
            # connect to cheapest visited node
            best = min(visited - {node}, key=lambda u: cprime(node, u))
            cost += cprime(node, best)
            degrees[node] += 1
            degrees[best] += 1
    # Add two cheapest depot edges
    depot_edges = []
    for client in oracle.nodes():
        if client == depot:
            continue
        depot_edges.append((cprime(depot, client), client))
    depot_edges.sort()
    for i in range(2):
        w, node = depot_edges[i]
        cost += w
        degrees[node] += 1
        degrees[depot] += 1
    return cost, degrees


def makespan_lower_bound(oracle: DistanceOracle, lb_tsp: float, k: int) -> float:
    depot = oracle.depot_id
    radial = max(2 * oracle.get(depot, node) for node in oracle.nodes() if node != depot)
    return max(lb_tsp / k, radial)
