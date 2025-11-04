from __future__ import annotations
import random
import statistics
from typing import List
from .data import Instance
from .sweep import sweep_build
from .tabu import TabuSearch
from .evaluator import solution_cost, eval_route

def _route_cost_if_feasible(inst: Instance, route: List[int], veh_type: int) -> tuple[float, bool]:
    """Helper retournant (cost, feasible) pour une route donnée."""
    eval_res = eval_route(inst, route, veh_type)
    return eval_res.cost, eval_res.feasible


def inter_routes_improvement(inst: Instance, routes: List[List[int]], veh_types: List[int]) -> bool:
    """Améliorations inter-routes via relocate / swap / 2-opt* en best-improvement."""

    if not routes:
        return False

    improved_once = False
    
    def _bounding_box(route: List[int]) -> tuple[float, float, float, float] | None:
        if not route:
            return None
        xs = [inst.coords[c][0] for c in route]
        ys = [inst.coords[c][1] for c in route]
        return (min(xs), max(xs), min(ys), max(ys))

    def _overlap(box_a: tuple[float, float, float, float] | None,
                 box_b: tuple[float, float, float, float] | None,
                 padding: float = 5.0) -> bool:
        if box_a is None or box_b is None:
            return True
        ax0, ax1, ay0, ay1 = box_a
        bx0, bx1, by0, by1 = box_b
        return not (ax1 + padding < bx0 or bx1 + padding < ax0 or
                    ay1 + padding < by0 or by1 + padding < ay0)

    def _candidate_positions(route: List[int], customer: int, k: int = 6) -> List[int]:
        if not route:
            return [0]
        candidates = []
        neighs = inst.candidate_lists[customer]
        for pos in range(len(route) + 1):
            prev_node = route[pos - 1] if pos > 0 else 0
            next_node = route[pos] if pos < len(route) else 0
            delta = (inst.dist[prev_node][customer] + inst.dist[customer][next_node]
                     - inst.dist[prev_node][next_node])
            if prev_node in neighs or next_node in neighs:
                candidates.append((delta, pos))
        if not candidates:
            for pos in range(len(route) + 1):
                prev_node = route[pos - 1] if pos > 0 else 0
                next_node = route[pos] if pos < len(route) else 0
                delta = (inst.dist[prev_node][customer] + inst.dist[customer][next_node]
                         - inst.dist[prev_node][next_node])
                candidates.append((delta, pos))
        candidates.sort(key=lambda x: x[0])
        return sorted(pos for _, pos in candidates[:k])


    improved = True
    while improved:
        improved = False
        best_move = None
        best_delta = -1e-6

        base_costs: List[float] = []
        bboxes: List[tuple[float, float, float, float] | None] = []
        for r_idx, route in enumerate(routes):
            cost, _ = _route_cost_if_feasible(inst, route, veh_types[r_idx])
            base_costs.append(cost)
            bboxes.append(_bounding_box(route))

        # Relocate moves: déplacer un client d'une route vers une autre
        for i in range(len(routes)):
            route_i = routes[i]
            if not route_i:
                continue
            cost_i = base_costs[i]
            veh_i = veh_types[i]

            for pos_i, customer in enumerate(route_i):
                new_route_i = route_i[:pos_i] + route_i[pos_i + 1 :]
                new_cost_i, feas_i = _route_cost_if_feasible(inst, new_route_i, veh_i)
                if not feas_i:
                    continue

                for j in range(len(routes)):
                    if i == j:
                        continue
                    route_j = routes[j]
                    veh_j = veh_types[j]
                    cost_j = base_costs[j]

                    if not _overlap(bboxes[i], bboxes[j]):
                        continue

                    insert_positions = _candidate_positions(route_j, customer)
                    for pos_j in insert_positions:
                        new_route_j = route_j[:pos_j] + [customer] + route_j[pos_j:]
                        new_cost_j, feas_j = _route_cost_if_feasible(inst, new_route_j, veh_j)
                        if not feas_j:
                            continue

                        delta = (new_cost_i + new_cost_j) - (cost_i + cost_j)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = ("relocate", i, j, new_route_i, new_route_j)

        # Swap moves: échanger deux clients entre deux routes différentes
        for i in range(len(routes)):
            route_i = routes[i]
            veh_i = veh_types[i]
            cost_i = base_costs[i]

            for j in range(i + 1, len(routes)):
                route_j = routes[j]
                veh_j = veh_types[j]
                cost_j = base_costs[j]

                if not route_i or not route_j:
                    continue
                
                if not _overlap(bboxes[i], bboxes[j]):
                    continue

                for pos_i, customer_i in enumerate(route_i):
                    for pos_j, customer_j in enumerate(route_j):
                        new_route_i = route_i[:pos_i] + [customer_j] + route_i[pos_i + 1 :]
                        new_route_j = route_j[:pos_j] + [customer_i] + route_j[pos_j + 1 :]

                        new_cost_i, feas_i = _route_cost_if_feasible(inst, new_route_i, veh_i)
                        if not feas_i:
                            continue
                        new_cost_j, feas_j = _route_cost_if_feasible(inst, new_route_j, veh_j)
                        if not feas_j:
                            continue

                        delta = (new_cost_i + new_cost_j) - (cost_i + cost_j)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = ("swap", i, j, new_route_i, new_route_j)
        # 2-opt* : échange des suffixes entre deux routes
        for i in range(len(routes)):
            route_i = routes[i]
            veh_i = veh_types[i]
            cost_i = base_costs[i]

            for j in range(i + 1, len(routes)):
                route_j = routes[j]
                veh_j = veh_types[j]
                cost_j = base_costs[j]

                len_i = len(route_i)
                len_j = len(route_j)
                
                if not _overlap(bboxes[i], bboxes[j]):
                    continue

                def _iter_cuts(length: int) -> List[int]:
                    if length <= 6:
                        return list(range(length + 1))
                    step = max(1, length // 6)
                    cuts = list(range(0, length + 1, step))
                    if cuts[-1] != length:
                        cuts.append(length)
                    if cuts[1] != 1:
                        cuts.append(1)
                    if length - 1 not in cuts:
                        cuts.append(length - 1)
                    return sorted(set(cuts))

                cuts_i = _iter_cuts(len_i)
                cuts_j = _iter_cuts(len_j)

                for cut_i in cuts_i:
                    prefix_i = route_i[:cut_i]
                    suffix_i = route_i[cut_i:]

                    for cut_j in cuts_j:
                        prefix_j = route_j[:cut_j]
                        suffix_j = route_j[cut_j:]

                        if not suffix_i and not suffix_j:
                            continue

                        new_route_i = prefix_i + suffix_j
                        new_route_j = prefix_j + suffix_i

                        new_cost_i, feas_i = _route_cost_if_feasible(inst, new_route_i, veh_i)
                        if not feas_i:
                            continue
                        new_cost_j, feas_j = _route_cost_if_feasible(inst, new_route_j, veh_j)
                        if not feas_j:
                            continue

                        delta = (new_cost_i + new_cost_j) - (cost_i + cost_j)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = ("2opt_star", i, j, new_route_i, new_route_j)
                            
        # Cross-exchange 2-2 : échange de deux arcs consécutifs entre routes voisines
        for i in range(len(routes)):
            route_i = routes[i]
            veh_i = veh_types[i]
            cost_i = base_costs[i]

            for j in range(i + 1, len(routes)):
                route_j = routes[j]
                veh_j = veh_types[j]
                cost_j = base_costs[j]

                if len(route_i) < 2 or len(route_j) < 2:
                    continue

                if not _overlap(bboxes[i], bboxes[j]):
                    continue

                for pos_i in range(len(route_i) - 1):
                    seg_i = route_i[pos_i : pos_i + 2]
                    neigh_i = set(inst.candidate_lists[seg_i[0]]) | set(inst.candidate_lists[seg_i[1]])
                    for pos_j in range(len(route_j) - 1):
                        seg_j = route_j[pos_j : pos_j + 2]
                        if seg_j[0] not in neigh_i and seg_j[1] not in neigh_i:
                            continue

                        options_i = {tuple(seg_i), tuple(reversed(seg_i))}
                        options_j = {tuple(seg_j), tuple(reversed(seg_j))}

                        for opt_i in options_i:
                            for opt_j in options_j:
                                new_route_i = route_i[:pos_i] + list(opt_j) + route_i[pos_i + 2 :]
                                new_route_j = route_j[:pos_j] + list(opt_i) + route_j[pos_j + 2 :]

                                new_cost_i, feas_i = _route_cost_if_feasible(inst, new_route_i, veh_i)
                                if not feas_i:
                                    continue
                                new_cost_j, feas_j = _route_cost_if_feasible(inst, new_route_j, veh_j)
                                if not feas_j:
                                    continue

                                delta = (new_cost_i + new_cost_j) - (cost_i + cost_j)
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = ("cross_exchange", i, j, new_route_i, new_route_j)
                           

        if best_move is not None:
            _move, i, j, new_route_i, new_route_j = best_move
            routes[i] = new_route_i
            routes[j] = new_route_j
            improved = True
            improved_once = True

    return improved_once

def solve_vrp(inst: Instance,
              rng_seed: int = 0,
              lamQ: float = 1000.0, lamT: float = 100.0,
              tabu_max_iter: int = 2000, tabu_no_improve: int = 200,
              seeds: List[int] | None = None,
              multi_seed_count: int = 3) -> dict:

    def _single_run(seed: int) -> dict:
        base_starts = 12
        if "X-n101-k25" in inst.name:
            base_starts = 24
        elif inst.n >= 90:
            base_starts = max(base_starts, 18)

        angle_jitter = 0.12 + 0.02 * (seed % 5)
        local_trials = 3
        pert_strength = 4

        routes, veh_types = sweep_build(
            inst,
            veh_type=0,
            num_starts=base_starts,
            rng_seed=seed,
            angle_jitter=angle_jitter,
            local_shuffle_trials=local_trials,
            perturbation_strength=pert_strength,
        )

        # 2) Route-Second (Tabu Search par route)
        adj_max_iter = tabu_max_iter
        adj_no_improve = tabu_no_improve
        if inst.n >= 80:
            adj_max_iter = max(adj_max_iter, 2500)
            adj_no_improve = max(adj_no_improve, 300)

        ts = TabuSearch(
            lamQ=lamQ,
            lamT=lamT,
            max_iter=adj_max_iter,
            no_improve_limit=adj_no_improve,
            rng_seed=seed,
        )
        for r in range(len(routes)):
            routes[r] = ts.improve_route(inst, routes[r], veh_type=veh_types[r])

        # 3) Raffinement inter-routes avec ré-optimisation intra-route
        for _ in range(3):
            improved = inter_routes_improvement(inst, routes, veh_types)
            if not improved:
                break
            for r in range(len(routes)):
                if routes[r]:
                    routes[r] = ts.improve_route(inst, routes[r], veh_type=veh_types[r])

        # Évaluation finale (sans pénalités)
        cost, dist = solution_cost(inst, routes, veh_types, lamQ=0.0, lamT=0.0)
        nb_veh = sum(1 for r in routes if len(r) > 0)
        route_evals = [eval_route(inst, r, veh_types[i]) for i, r in enumerate(routes)]
        feas = all(ev.feasible for ev in route_evals)

        depot_open = getattr(inst, "depot_open", 0.0)
        route_end_times = [ev.time_end for ev in route_evals]
        route_durations = [max(0.0, end - depot_open) for end in route_end_times]
        makespan = max(route_durations) if route_durations else 0.0

        return {
            "seed": seed,
            "routes": [r[:] for r in routes],
            "veh_types": veh_types[:],
            "cost": cost,
            "distance": dist,
            "used_vehicles": nb_veh,
            "feasible": feas,
            "route_end_times": route_end_times,
            "route_durations": route_durations,
            "makespan": makespan,
        }

    if seeds is None or not seeds:
        rng = random.Random(rng_seed)
        count = max(1, multi_seed_count)
        seeds = []
        for idx in range(count):
            if idx == 0 and rng_seed is not None:
                seeds.append(rng_seed)
            else:
                seeds.append(rng.randint(0, 10**6))

    run_results: List[dict] = []
    best_idx: int | None = None

    for seed in seeds:
        res = _single_run(seed)
        run_results.append(res)
        if best_idx is None or res["cost"] < run_results[best_idx]["cost"]:
            best_idx = len(run_results) - 1

    assert best_idx is not None  # pour mypy
    best_run = run_results[best_idx]

    costs = [res["cost"] for res in run_results]
    avg_cost = statistics.mean(costs) if costs else 0.0
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0

    return {
        "routes": [r[:] for r in best_run["routes"]],
        "veh_types": best_run["veh_types"][:],
        "cost": best_run["cost"],
        "distance": best_run["distance"],
        "used_vehicles": best_run["used_vehicles"],
        "feasible": best_run["feasible"],
        "best_seed": best_run["seed"],
        "route_end_times": best_run.get("route_end_times", []),
        "route_durations": best_run.get("route_durations", []),
        "makespan": best_run.get("makespan", 0.0),
        "runs": run_results,
        "stats": {
            "mean_cost": avg_cost,
            "stdev_cost": std_cost,
            "num_runs": len(run_results),
        },
    }