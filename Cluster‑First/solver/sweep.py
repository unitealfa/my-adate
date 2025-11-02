from __future__ import annotations
from typing import List, Tuple
from math import atan2
import random

from .data import Instance
from .evaluator import eval_route, solution_cost

def _build_routes_from_order(inst: Instance, order: List[int], veh_type: int) -> Tuple[List[List[int]], List[int]]:
    routes: List[List[int]] = []
    veh_types: List[int] = []
    current: List[int] = []

    for cid in order:
        trial = current + [cid]
        er = eval_route(inst, trial, veh_type)
        if er.capacity_excess == 0.0 and er.tw_violation == 0.0 and er.time_end <= inst.depot_close:
            current.append(cid)
        else:
            if current:
                routes.append(current)
                veh_types.append(veh_type)
            current = [cid]

    if current:
        routes.append(current)
        veh_types.append(veh_type)

    return routes, veh_types


def sweep_build(
    inst: Instance,
    veh_type: int = 0,
    num_starts: int = 8,
    rng_seed: int | None = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Sweep capacitaire + TW: tri par angle autour du dépôt, empilement séquentiel
    avec test de faisabilité (capacité + TW). Retourne (routes, veh_types).
    """
    coords = inst.coords
    n = inst.n

    # Calcul des angles polaires autour du dépôt
    angles = []
    for i in range(1, n + 1):
        dx = coords[i][0] - coords[0][0]
        dy = coords[i][1] - coords[0][1]
        angles.append((i, atan2(dy, dx)))
    angles.sort(key=lambda x: x[1])

    order = [cid for cid, _ in angles]
    if not order:
        return [], []

    rng = random.Random(rng_seed)
    starts: List[int] = []
    if num_starts >= len(order):
        starts = list(range(len(order)))
    else:
        starts = [0]
        while len(starts) < num_starts:
            cand = rng.randrange(len(order))
            if cand not in starts:
                starts.append(cand)
        starts.sort()

    best_routes: List[List[int]] | None = None
    best_veh: List[int] | None = None
    best_cost = float("inf")

    for offset in starts:
        rotated = order[offset:] + order[:offset]
        routes, veh_types = _build_routes_from_order(inst, rotated, veh_type)
        cost, _ = solution_cost(inst, routes, veh_types, lamQ=0.0, lamT=0.0)
        if cost < best_cost:
            best_cost = cost
            best_routes = [r[:] for r in routes]
            best_veh = veh_types[:]

        # Aussi essayer l'ordre inverse pour diversifier
        rotated_rev = list(reversed(rotated))
        routes_rev, veh_types_rev = _build_routes_from_order(inst, rotated_rev, veh_type)
        cost_rev, _ = solution_cost(inst, routes_rev, veh_types_rev, lamQ=0.0, lamT=0.0)
        if cost_rev < best_cost:
            best_cost = cost_rev
            best_routes = [r[:] for r in routes_rev]
            best_veh = veh_types_rev[:]

    if best_routes is None or best_veh is None:
        return [], []

    return best_routes, best_veh