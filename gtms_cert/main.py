"""CLI entry point implementing the GTMS-Cert pipeline."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence, Tuple, cast

from .improve import improve_makespan
from .io import (
    ProblemData,
    build_output_payload,
    generate_random_problem,
    read_input,
    write_output,
)
from .lb import held_karp_1tree_lb, makespan_lower_bound
from .split import minimax_split
from .tsp import build_giant_tour
from .utils import configure_logging, log_progress, set_seed


def solve_problem_data(
    data: ProblemData,
    *,
    seed: int = 42,
    lb_iters: int = 200,
) -> Tuple[List[List[str]], float, float, float, int]:
    """Run the GTMS-Cert pipeline on an in-memory problem description."""
    configure_logging()
    set_seed(seed)
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
    return routes, ub, lb, gap, longest_route_idx


def solve_gtms_cert(
    input_json_path: str,
    output_json_path: str,
    seed: int = 42,
    cands: int = 32,
    lb_iters: int = 200,
) -> Dict[str, object]:
    """Solve the mTSP instance stored in JSON format using the GTMS-Cert pipeline."""
    data = read_input(input_json_path, cands=cands)
    routes, ub, lb, gap, longest_route_idx = solve_problem_data(
        data,
        seed=seed,
        lb_iters=lb_iters,
    )
    return write_output(output_json_path, routes, data.oracle, ub, lb, gap, longest_route_idx)


def _parts_to_routes(parts: List[tuple[int, int]], tour: List[str], depot: str) -> List[List[str]]:
    routes: List[List[str]] = []
    for start, end in parts:
        segment = [depot] + tour[start : end + 1] + [depot]
        routes.append(segment)
    return routes


def _prompt_int(
    prompt: str,
    *,
    minimum: int = 0,
    default: int | None = None,
    message_suffix: str = "",
) -> int:
    """Prompt the user for an integer value with optional validation."""
    while True:
        message = prompt
        if message_suffix:
            message += message_suffix
        if default is not None:
            message += f" [{default}]"
        message += ": "
        raw = input(message).strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Veuillez entrer un nombre entier valide.")
                continue
        if value < minimum:
            print(f"Veuillez saisir une valeur supérieure ou égale à {minimum}.")
            continue
        return value


def _print_solution_summary(seed: int, data: ProblemData, payload: Dict[str, object]) -> None:
    """Display the optimisation result in a human-friendly format."""
    print("\n=== Instance générée ===")
    print(f"Seed utilisée : {seed}")
    print(f"Nombre de camions : {data.vehicle_count}")
    print(f"Nombre de clients : {len(data.clients)}")

    print("\n=== Résultat de l'optimisation ===")
    routes = cast(Sequence[Dict[str, Any]], payload.get("routes", []))
    for route in routes:
        vehicle = route.get("vehicle")
        sequence = cast(Sequence[str], route.get("sequence", []))
        time_min = float(route.get("time_min", 0.0))
        sequence_str = " -> ".join(sequence)
        print(f"Camion {vehicle}: {sequence_str} | Durée : {time_min:.2f} minutes")

    makespan = float(payload.get("makespan", 0.0))
    longest_vehicle = payload.get("longest_route_vehicle")
    lower_bound = float(payload.get("best_lower_bound", 0.0))
    gap_percent = float(payload.get("gap", 0.0)) * 100
    print(
        f"\nDurée maximale de tournée : {makespan:.2f} minutes (Camion {longest_vehicle})"
    )
    print(f"Borne inférieure : {lower_bound:.2f} minutes")
    print(f"Écart relatif : {gap_percent:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="GTMS-Cert solver")
    parser.add_argument("--input", help="Input JSON path")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--seed", type=int, help="Seed to use for deterministic runs")
    parser.add_argument("--cands", type=int, default=32)
    parser.add_argument("--lb-iters", type=int, default=200)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate a random instance interactively (default when no input is provided)",
    )
    args = parser.parse_args()

    if args.input and args.interactive:
        parser.error("--interactive cannot be combined with --input")

    if args.input:
        if not args.output:
            parser.error("--output is required when --input is provided")
        seed = args.seed if args.seed is not None else 42
        solve_gtms_cert(
            args.input,
            args.output,
            seed=seed,
            cands=args.cands,
            lb_iters=args.lb_iters,
        )
        return

    trucks = _prompt_int("Nombre de camions disponibles", minimum=1)
    clients = _prompt_int(
        "Nombre de clients",
        minimum=trucks,
        message_suffix=" (>= nombre de camions)",
    )
    seed_default = args.seed if args.seed is not None else 42
    interactive_seed = _prompt_int("Seed à utiliser", minimum=0, default=seed_default)

    data = generate_random_problem(
        trucks,
        clients,
        interactive_seed,
        cands=args.cands,
    )
    routes, ub, lb, gap, longest_idx = solve_problem_data(
        data,
        seed=interactive_seed,
        lb_iters=args.lb_iters,
    )
    payload = build_output_payload(routes, data.oracle, ub, lb, gap, longest_idx)
    _print_solution_summary(interactive_seed, data, payload)

    if args.output:
        write_output(args.output, routes, data.oracle, ub, lb, gap, longest_idx)


if __name__ == "__main__":
    main()
