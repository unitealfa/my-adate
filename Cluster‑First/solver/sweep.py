from __future__ import annotations

import random
from math import atan2
from typing import List, Tuple

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
    angle_jitter: float = 0.15,
    local_shuffle_trials: int = 2,
    perturbation_strength: int = 3,
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
    angle_map = {cid: ang for cid, ang in angles}
    if not order:
        return [], []

    rng = random.Random(rng_seed)
    starts: List[int]
    if num_starts >= len(order):
        starts = list(range(len(order)))
    else:
        starts = [0]
        while len(starts) < num_starts:
            cand = rng.randrange(len(order))
            if cand not in starts:
                starts.append(cand)
        starts.sort()
    tested_orders: List[List[int]] = []
    seen_orders: set[Tuple[int, ...]] = set()

    def _register(candidate: List[int]) -> None:
        key = tuple(candidate)
        if not candidate or key in seen_orders:
            return
        seen_orders.add(key)
        tested_orders.append(candidate)

    def _local_shuffles(base: List[int]) -> None:
        if len(base) < 3:
            return
        for _ in range(local_shuffle_trials):
            perm = base[:]
            swaps = min(perturbation_strength, max(1, len(perm) - 1))
            for _ in range(swaps):
                idx = rng.randrange(len(perm) - 1)
                perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
            _register(perm)

    for offset in starts:
        rotated = order[offset:] + order[:offset]
        _register(rotated)
        _local_shuffles(rotated)
        rotated_rev = list(reversed(rotated))
        _register(rotated_rev)
        _local_shuffles(rotated_rev)

    jitter_trials = max(1, min(len(order), num_starts // 2))
    for _ in range(jitter_trials):
        jittered_pairs = [
            (cid, angle_map[cid] + rng.uniform(-angle_jitter, angle_jitter))
            for cid in order
        ]
        jittered_pairs.sort(key=lambda x: x[1])
        jittered = [cid for cid, _ in jittered_pairs]
        _register(jittered)
        _local_shuffles(jittered)

    best_routes: List[List[int]] | None = None
    best_veh: List[int] | None = None
    best_cost = float("inf")

    for candidate in tested_orders:
        routes, veh_types = _build_routes_from_order(inst, candidate, veh_type)
        cost, _ = solution_cost(inst, routes, veh_types, lamQ=0.0, lamT=0.0)
        if cost < best_cost:
            best_cost = cost
            best_routes = [r[:] for r in routes]
            best_veh = veh_types[:]

    if best_routes is None or best_veh is None:
        return [], []

    return best_routes, best_veh