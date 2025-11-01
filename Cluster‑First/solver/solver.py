from __future__ import annotations
from typing import List
from .data import Instance
from .sweep import sweep_build
from .tabu import TabuSearch
from .evaluator import solution_cost, eval_route

def _route_cost_if_feasible(inst: Instance, route: List[int], veh_type: int) -> tuple[float, bool]:
    """Helper retournant (cost, feasible) pour une route donnée."""
    eval_res = eval_route(inst, route, veh_type)
    return eval_res.cost, eval_res.feasible


def inter_routes_improvement(inst: Instance, routes: List[List[int]], veh_types: List[int]) -> None:
    """Simple amélioration entre routes via relocate / swap best-improvement."""

    if not routes:
        return

    improved = True
    while improved:
        improved = False
        best_move = None
        best_delta = -1e-6

        base_costs: List[float] = []
        for r_idx, route in enumerate(routes):
            cost, _ = _route_cost_if_feasible(inst, route, veh_types[r_idx])
            base_costs.append(cost)

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

                    for pos_j in range(len(route_j) + 1):
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

        if best_move is not None:
            move_type, i, j, new_route_i, new_route_j = best_move
            routes[i] = new_route_i
            routes[j] = new_route_j
            improved = True

def solve_vrp(inst: Instance,
              rng_seed: int = 0,
              lamQ: float = 1000.0, lamT: float = 100.0,
              tabu_max_iter: int = 2000, tabu_no_improve: int = 200):
    # 1) Cluster-First (Sweep)
    routes, veh_types = sweep_build(inst, veh_type=0)

    # 2) Route-Second (Tabu Search par route)
    ts = TabuSearch(lamQ=lamQ, lamT=lamT, max_iter=tabu_max_iter,
                    no_improve_limit=tabu_no_improve, rng_seed=rng_seed)
    for r in range(len(routes)):
        routes[r] = ts.improve_route(inst, routes[r], veh_type=veh_types[r])

    # 3) Raffinement inter-routes (optionnel)
    inter_routes_improvement(inst, routes, veh_types)

    # Évaluation finale (sans pénalités)
    cost, dist = solution_cost(inst, routes, veh_types, lamQ=0.0, lamT=0.0)
    nb_veh = sum(1 for r in routes if len(r) > 0)
    feas = all(eval_route(inst, r, veh_types[i]).feasible for i, r in enumerate(routes))

    return {
        "routes": routes,
        "veh_types": veh_types,
        "cost": cost,
        "distance": dist,
        "used_vehicles": nb_veh,
        "feasible": feas,
    }
