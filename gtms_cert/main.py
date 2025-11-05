"""CLI entry point implementing the GTMS-Cert pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .improve import improve_makespan
from .io import ProblemData, read_input, write_output
from .lb import held_karp_1tree_lb, makespan_lower_bound
from .split import minimax_split
from .tsp import build_giant_tour
from .utils import configure_logging, log_progress, set_seed


def solve_gtms_cert(
    input_json_path: str,
    output_json_path: str,
    seed: int = 42,
    cands: int = 32,
    lb_iters: int = 200,
) -> None:
    """Solve the mTSP instance using the GTMS-Cert pipeline."""
    configure_logging()
    set_seed(seed)
    data = read_input(input_json_path, cands=cands)
    if not data.use_all:
        raise ValueError("Solver requires use_all=true")
    if len(data.clients) < data.vehicle_count:
        raise ValueError("Number of clients must be >= number of vehicles")
    oracle = data.oracle
    tour = build_giant_tour(data.clients, oracle, seed=seed)
    parts, ub = minimax_split(tour, data.depot_id, oracle, data.vehicle_count)
    routes = _parts_to_routes(parts, tour, data.depot_id)
    routes = improve_makespan(routes, oracle)
    costs = [oracle.route_time(r) for r in routes]
    ub = max(costs)
    lb_tsp = held_karp_1tree_lb(oracle, iterations=lb_iters)
    lb = makespan_lower_bound(oracle, lb_tsp, data.vehicle_count)
    gap = (ub - lb) / ub if ub > 0 else 0.0
    log_progress(ub, lb, gap)
    while gap > 0.01:
        routes = improve_makespan(routes, oracle)
        costs = [oracle.route_time(r) for r in routes]
        ub = max(costs)
        lb_tsp = max(lb_tsp, held_karp_1tree_lb(oracle, iterations=50))
        lb = makespan_lower_bound(oracle, lb_tsp, data.vehicle_count)
        gap = (ub - lb) / ub if ub > 0 else 0.0
        log_progress(ub, lb, gap)
    longest_route_idx = max(range(len(routes)), key=lambda idx: oracle.route_time(routes[idx]))
    write_output(output_json_path, routes, oracle, ub, lb, gap, longest_route_idx)


def _parts_to_routes(parts: List[tuple[int, int]], tour: List[str], depot: str) -> List[List[str]]:
    routes: List[List[str]] = []
    for start, end in parts:
        segment = [depot] + tour[start : end + 1] + [depot]
        routes.append(segment)
    return routes


def main() -> None:
    parser = argparse.ArgumentParser(description="GTMS-Cert solver")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cands", type=int, default=32)
    parser.add_argument("--lb-iters", type=int, default=200)
    args = parser.parse_args()
    solve_gtms_cert(args.input, args.output, seed=args.seed, cands=args.cands, lb_iters=args.lb_iters)


if __name__ == "__main__":
    main()
