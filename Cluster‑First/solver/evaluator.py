from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from .data import Instance

@dataclass
class RouteEval:
    feasible: bool
    cost: float
    distance: float
    time_end: float
    load: float
    waiting_total: float
    tw_violation: float
    capacity_excess: float

def eval_route(instance: Instance, route: List[int], veh_type: int = 0) -> RouteEval:
    """
    route = liste d'indices de clients (sans le dépôt).
    On ajoute dépôt (0) au début et à la fin pour l’évaluation.
    """
    cap = instance.capacity[veh_type]
    ckm = instance.cost_per_km[veh_type]
    fcx = instance.fixed_cost[veh_type]

    seq = [0] + route + [0]
    t = instance.depot_open
    load = 0.0
    waiting = 0.0
    tw_violation = 0.0
    dist = 0.0

    # Distance totale
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        dist += instance.dist[a][b]

    # Propagation temporelle (forward pass)
    t = instance.depot_open
    for idx in range(1, len(seq)):
        i_prev = seq[idx - 1]
        i = seq[idx]
        travel = instance.time[i_prev][i]
        t = t + travel

        # Client (pas dépôt final)
        if idx < len(seq) - 1:
            load += instance.demand[i]
            # Attente si on arrive avant le début de fenêtre
            if t < instance.window_a[i]:
                waiting += (instance.window_a[i] - t)
                t = instance.window_a[i]
            # Violation si on dépasse la fin de fenêtre
            if t > instance.window_b[i]:
                tw_violation += (t - instance.window_b[i])
            # Service
            t += instance.service[i]

    capacity_excess = max(0.0, load - cap)
    is_used = 1 if len(route) > 0 else 0
    cost = is_used * (fcx + ckm * dist)
    feasible = (capacity_excess == 0.0) and (tw_violation == 0.0) and (t <= instance.depot_close)

    return RouteEval(
        feasible=feasible,
        cost=cost,
        distance=dist,
        time_end=t,
        load=load,
        waiting_total=waiting,
        tw_violation=tw_violation,
        capacity_excess=capacity_excess,
    )

def route_cost_with_penalties(inst: Instance, route: List[int], veh_type: int,
                              lamQ: float, lamT: float) -> float:
    r = eval_route(inst, route, veh_type)
    return r.cost + lamQ * r.capacity_excess + lamT * r.tw_violation

def solution_cost(inst: Instance, routes: List[List[int]], veh_types: List[int],
                  lamQ: float, lamT: float) -> tuple[float, float]:
    tot = 0.0
    tot_dist = 0.0
    for r, k in zip(routes, veh_types):
        er = eval_route(inst, r, k)
        tot += er.cost + lamQ * er.capacity_excess + lamT * er.tw_violation
        tot_dist += er.distance
    return tot, tot_dist
