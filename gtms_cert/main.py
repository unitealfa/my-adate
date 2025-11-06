"""CLI entry point implementing the GTMS-Cert pipeline."""
from __future__ import annotations

import argparse
import logging
import random
from typing import Any, Dict, List, Sequence, Tuple, cast

from .geo import DistanceOracle
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
    max_gap_iterations: int = 100,
    stagnation_tolerance: int = 5,
    max_diversifications: int = 3,
    diversification_tolerance: float = 0.05,
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
    memory: List[Tuple[List[List[str]], float, float, float]] = []
    best_gap_seen = float("inf")
    logger = logging.getLogger("gtms_cert")
    rng = random.Random(seed)
    diversification_attempts = 0
    if gap < best_gap_seen:
        memory.append(([route[:] for route in routes], ub, lb, gap))
        best_gap_seen = gap
        logger.info("État initial ajouté en mémoire (gap %.2f%%).", gap * 100)
    iterations = 0
    stagnant_steps = 0
    previous_ub = ub
    previous_gap = gap
    repeated_gap_steps = 0
    while gap > 0.01 and iterations < max_gap_iterations:
        iterations += 1
        routes = improve_makespan(routes, oracle)
        costs = [oracle.route_time(r) for r in routes]
        ub = max(costs)
        lb_tsp = max(lb_tsp, held_karp_1tree_lb(oracle, iterations=50))
        lb = makespan_lower_bound(oracle, lb_tsp, data.vehicle_count)
        gap = (ub - lb) / ub if ub > 0 else 0.0
        log_progress(ub, lb, gap)
        if gap + 1e-9 < best_gap_seen:
            memory.append(([route[:] for route in routes], ub, lb, gap))
            best_gap_seen = gap
            if len(memory) > 10:
                memory.pop(0)
            logger.info("Nouvel état stocké en mémoire (gap %.2f%%).", gap * 100)
            diversification_attempts = 0
        if abs(ub - previous_ub) < 1e-6:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        previous_ub = ub
        if abs(gap - previous_gap) < 1e-6:
            repeated_gap_steps += 1
        else:
            repeated_gap_steps = 0
        previous_gap = gap
        if repeated_gap_steps >= 5:
            repeated_gap_steps = 0
            stagnant_steps = 0
            used_memory = False
            improved = False
            if memory:
                best_index = min(range(len(memory)), key=lambda idx: memory[idx][3])
                best_routes, best_ub, best_lb, best_gap = memory.pop(best_index)
                routes = [route[:] for route in best_routes]
                ub = best_ub
                lb = best_lb
                gap = best_gap
                used_memory = True
                logger.info(
                    "Gap stagnant détecté : utilisation de la mémoire (gap %.2f%%) pour relancer l'optimisation.",
                    gap * 100,
                )
                routes = improve_makespan(routes, oracle)
                costs = [oracle.route_time(r) for r in routes]
                ub = max(costs)
                lb_tsp = max(lb_tsp, held_karp_1tree_lb(oracle, iterations=50))
                lb = makespan_lower_bound(oracle, lb_tsp, data.vehicle_count)
                gap = (ub - lb) / ub if ub > 0 else 0.0
                log_progress(ub, lb, gap)
                if gap + 1e-9 < best_gap_seen:
                    memory.append(([route[:] for route in routes], ub, lb, gap))
                    best_gap_seen = gap
                    if len(memory) > 10:
                        memory.pop(0)
                    logger.info(
                        "État optimisé depuis la mémoire ajouté (gap %.2f%%).",
                        gap * 100,
                    )
                    diversification_attempts = 0
                    improved = True
                else:
                    logger.info("Redémarrage depuis la mémoire sans amélioration notable.")
            diversification_performed = False
            if not improved and diversification_attempts < max_diversifications:
                diversification_attempts += 1
                diversification_performed = _diversify_routes(
                    routes,
                    oracle,
                    rng,
                    tolerance=diversification_tolerance,
                )
                if diversification_performed:
                    logger.info(
                        "Diversification appliquée après stagnation (tentative %d).",
                        diversification_attempts,
                    )
                    routes = improve_makespan(routes, oracle)
                    costs = [oracle.route_time(r) for r in routes]
                    ub = max(costs)
                    lb_tsp = max(lb_tsp, held_karp_1tree_lb(oracle, iterations=50))
                    lb = makespan_lower_bound(oracle, lb_tsp, data.vehicle_count)
                    gap = (ub - lb) / ub if ub > 0 else 0.0
                    log_progress(ub, lb, gap)
                    if gap + 1e-9 < best_gap_seen:
                        memory.append(([route[:] for route in routes], ub, lb, gap))
                        best_gap_seen = gap
                        if len(memory) > 10:
                            memory.pop(0)
                        logger.info(
                            "État diversifié ajouté en mémoire (gap %.2f%%).",
                            gap * 100,
                        )
                        diversification_attempts = 0
                        improved = True
                else:
                    logger.info(
                        "Échec de la diversification aléatoire (tentative %d).",
                        diversification_attempts,
                    )
            elif not improved and diversification_attempts >= max_diversifications:
                logger.info(
                    "Nombre maximal de tentatives de diversification atteint (%d).",
                    max_diversifications,
                )
            previous_ub = ub
            previous_gap = gap
            if improved or diversification_performed or used_memory:
                continue
            logger.info("Aucune stratégie de diversification disponible après stagnation ; arrêt anticipé.")
            break
        if stagnant_steps >= stagnation_tolerance:
            logger.info(
                "Arrêt de l'amélioration : aucune réduction du makespan après %d itérations.",
                stagnant_steps,
            )
            break
    if iterations >= max_gap_iterations and gap > 0.01:
        logging.getLogger("gtms_cert").info(
            "Arrêt de l'amélioration après %d itérations (gap final %.2f%%).",
            iterations,
            gap * 100,
        )
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


def _diversify_routes(
    routes: List[List[str]],
    oracle: DistanceOracle,
    rng: random.Random,
    *,
    attempts: int = 40,
    tolerance: float = 0.05,
) -> bool:
    """Introduce a controlled random perturbation to escape stagnation."""

    if not routes:
        return False
    current_makespan = max(oracle.route_time(r) for r in routes)
    for attempt in range(max(1, attempts)):
        longest_idx = max(range(len(routes)), key=lambda idx: oracle.route_time(routes[idx]))
        longest_route = routes[longest_idx]
        if len(longest_route) <= 3:
            return False
        node_pos = rng.randrange(1, len(longest_route) - 1)
        node = longest_route[node_pos]
        destination_candidates = [idx for idx in range(len(routes)) if idx != longest_idx]
        if not destination_candidates:
            return False
        dest_idx = rng.choice(destination_candidates)
        dest_route = routes[dest_idx]
        insert_pos = rng.randrange(1, len(dest_route))
        new_routes = [list(r) for r in routes]
        new_longest = longest_route[:node_pos] + longest_route[node_pos + 1 :]
        if len(new_longest) < 3:
            continue
        new_dest = dest_route[:insert_pos] + [node] + dest_route[insert_pos:]
        new_routes[longest_idx] = new_longest
        new_routes[dest_idx] = new_dest
        new_makespan = max(oracle.route_time(r) for r in new_routes)
        dynamic_tolerance = tolerance * (1 + attempt // 10)
        if new_makespan <= current_makespan * (1 + dynamic_tolerance):
            routes.clear()
            routes.extend(new_routes)
            return True
    return False


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


def print_solution_summary(seed: int, data: ProblemData, payload: Dict[str, object]) -> None:
    """Display the optimisation result in a human-friendly format."""
    vehicle_count = data.vehicle_count
    client_count = len(data.clients)
    gap_percent = float(payload.get("gap", 0.0)) * 100
    lower_bound = float(payload.get("best_lower_bound", 0.0))
    makespan = float(payload.get("makespan", 0.0))

    print("\n=== Paramètres de l'instance ===")
    print(f"Seed utilisée : {seed}")
    print(f"Nombre de camions : {vehicle_count}")
    print(f"Nombre de clients : {client_count}")

    print("\n=== Détails des tournées ===")
    routes = cast(Sequence[Dict[str, Any]], payload.get("routes", []))
    route_durations: List[tuple[int, float]] = []
    for route in routes:
        vehicle = int(route.get("vehicle", 0))
        sequence = cast(Sequence[str], route.get("sequence", []))
        time_min = float(route.get("time_min", 0.0))
        sequence_str = " -> ".join(sequence)
        print(f"Camion {vehicle}: {sequence_str} | Durée : {time_min:.2f} minutes")
        route_durations.append((vehicle, time_min))

    longest_vehicle = payload.get("longest_route_vehicle")
    longest_time = makespan
    if route_durations:
        vehicle_longest, duration_longest = max(route_durations, key=lambda item: item[1])
        longest_time = duration_longest
        if longest_vehicle in (None, 0):
            longest_vehicle = vehicle_longest

    print("\n=== Synthèse finale ===")
    print(f"Gap final : {gap_percent:.2f}%")
    print(f"Borne inférieure : {lower_bound:.2f} minutes")
    print(
        f"Temps complet du dernier camion : {longest_time:.2f} minutes",
    )
    print(
        f"Tournée la plus longue : Camion {longest_vehicle} ({longest_time:.2f} minutes)",
    )


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

    default_trucks = 5
    trucks = _prompt_int(
        "Nombre de camions disponibles",
        minimum=1,
        default=default_trucks,
        message_suffix=" (obligatoire — Entrée pour valeur par défaut)",
    )
    clients = _prompt_int(
        "Nombre de clients",
        minimum=trucks,
        default=max(trucks, 20),
        message_suffix=" (>= nombre de camions, obligatoire — Entrée pour valeur par défaut)",
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
    print_solution_summary(interactive_seed, data, payload)

    if args.output:
        write_output(args.output, routes, data.oracle, ub, lb, gap, longest_idx)


if __name__ == "__main__":
    main()
