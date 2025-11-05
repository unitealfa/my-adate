"""Local improvement heuristics for the minimax objective."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .geo import DistanceOracle


def improve_makespan(routes: List[List[str]], oracle: DistanceOracle, max_loops: int = 20) -> List[List[str]]:
    """Iteratively improve the makespan using relocate, swap and 2-opt*."""
    if not routes:
        return routes
    for _ in range(max_loops):
        improved = False
        if _relocate(routes, oracle):
            improved = True
        if _swap(routes, oracle):
            improved = True
        if _two_opt_star(routes, oracle):
            improved = True
        if not improved:
            break
    return routes


def _makespan(routes: Sequence[Sequence[str]], oracle: DistanceOracle) -> Tuple[float, int]:
    costs = [oracle.route_time(r) for r in routes]
    max_cost = max(costs)
    idx = max(range(len(routes)), key=lambda i: costs[i])
    return max_cost, idx


def _relocate(routes: List[List[str]], oracle: DistanceOracle) -> bool:
    current_makespan, longest_idx = _makespan(routes, oracle)
    longest_route = routes[longest_idx]
    if len(longest_route) <= 3:
        return False
    for pos in range(1, len(longest_route) - 1):
        if len(longest_route) <= 3:
            break
        node = longest_route[pos]
        prev_node = longest_route[pos - 1]
        next_node = longest_route[pos + 1]
        removal_delta = (
            oracle.get(prev_node, next_node)
            - oracle.get(prev_node, node)
            - oracle.get(node, next_node)
        )
        for ridx, route in enumerate(routes):
            if ridx == longest_idx:
                continue
            for insert_pos in range(1, len(route)):
                before = route[insert_pos - 1]
                after = route[insert_pos]
                insertion_delta = (
                    oracle.get(before, node)
                    + oracle.get(node, after)
                    - oracle.get(before, after)
                )
                new_routes = [list(r) for r in routes]
                candidate_longest = longest_route[:pos] + longest_route[pos + 1 :]
                if len(candidate_longest) < 3:
                    continue
                candidate_route = route[:insert_pos] + [node] + route[insert_pos:]
                new_routes[longest_idx] = candidate_longest
                new_routes[ridx] = candidate_route
                ms, _ = _makespan(new_routes, oracle)
                if ms + 1e-6 < current_makespan:
                    routes[longest_idx] = candidate_longest
                    routes[ridx] = candidate_route
                    return True
    return False


def _swap(routes: List[List[str]], oracle: DistanceOracle) -> bool:
    current_makespan, longest_idx = _makespan(routes, oracle)
    longest_route = routes[longest_idx]
    for pos_a in range(1, len(longest_route) - 1):
        node_a = longest_route[pos_a]
        prev_a = longest_route[pos_a - 1]
        next_a = longest_route[pos_a + 1]
        for ridx, route in enumerate(routes):
            if ridx == longest_idx or len(route) <= 3:
                continue
            for pos_b in range(1, len(route) - 1):
                node_b = route[pos_b]
                prev_b = route[pos_b - 1]
                next_b = route[pos_b + 1]
                delta_long = (
                    oracle.get(prev_a, node_b)
                    + oracle.get(node_b, next_a)
                    - oracle.get(prev_a, node_a)
                    - oracle.get(node_a, next_a)
                )
                delta_route = (
                    oracle.get(prev_b, node_a)
                    + oracle.get(node_a, next_b)
                    - oracle.get(prev_b, node_b)
                    - oracle.get(node_b, next_b)
                )
                if delta_long + delta_route >= 1e-9:
                    continue
                new_routes = [list(r) for r in routes]
                new_routes[longest_idx] = longest_route[:pos_a] + [node_b] + longest_route[pos_a + 1 :]
                new_routes[ridx] = route[:pos_b] + [node_a] + route[pos_b + 1 :]
                ms, _ = _makespan(new_routes, oracle)
                if ms + 1e-6 < current_makespan:
                    routes[longest_idx] = new_routes[longest_idx]
                    routes[ridx] = new_routes[ridx]
                    return True
    return False


def _two_opt_star(routes: List[List[str]], oracle: DistanceOracle) -> bool:
    current_makespan, longest_idx = _makespan(routes, oracle)
    longest_route = routes[longest_idx]
    for ridx, route in enumerate(routes):
        if ridx == longest_idx:
            continue
        for i in range(1, len(longest_route) - 1):
            a = longest_route[i - 1]
            b = longest_route[i]
            for j in range(1, len(route) - 1):
                c = route[j - 1]
                d = route[j]
                gain = (
                    oracle.get(a, d)
                    + oracle.get(c, b)
                    - oracle.get(a, b)
                    - oracle.get(c, d)
                )
                if gain >= -1e-6:
                    continue
                new_longest = longest_route[:i] + route[j:]
                new_route = route[:j] + longest_route[i:]
                if len(new_longest) < 3 or len(new_route) < 3:
                    continue
                new_routes = [list(r) for r in routes]
                new_routes[longest_idx] = new_longest
                new_routes[ridx] = new_route
                ms, _ = _makespan(new_routes, oracle)
                if ms + 1e-6 < current_makespan:
                    routes[longest_idx] = new_longest
                    routes[ridx] = new_route
                    return True
    return False
