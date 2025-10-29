from __future__ import annotations
from typing import List
from .data import Instance
from .sweep import sweep_build
from .tabu import TabuSearch
from .evaluator import solution_cost, eval_route

def inter_routes_improvement(inst: Instance, routes: List[List[int]], veh_types: List[int]) -> None:
    """
    Hook optionnel : améliorer entre routes (relocate/swap inter-routes).
    Laisse vide pour l’instant (simple), à compléter si tu veux gratter du coût.
    """
    return

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
