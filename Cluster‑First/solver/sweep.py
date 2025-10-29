from __future__ import annotations
from typing import List, Tuple
from math import atan2
import numpy as np

from .data import Instance
from .evaluator import eval_route

def sweep_build(inst: Instance, veh_type: int = 0) -> Tuple[List[List[int]], List[int]]:
    """
    Sweep capacitaire + TW: tri par angle autour du dépôt, empilement séquentiel
    avec test de faisabilité (capacité + TW). Retourne (routes, veh_types).
    """
    coords = inst.coords
    n = inst.n

    # Calcul des angles polaires autour du dépôt
    angles = []
    for i in range(1, n + 1):
        dx = coords[i, 0] - coords[0, 0]
        dy = coords[i, 1] - coords[0, 1]
        angles.append((i, atan2(dy, dx)))
    angles.sort(key=lambda x: x[1])

    routes: List[List[int]] = []
    veh_types: List[int] = []

    current: List[int] = []
    for cid, _ang in angles:
        # tentative d'insertion en fin de route
        trial = current + [cid]
        er = eval_route(inst, trial, veh_type)
        if er.capacity_excess == 0.0 and er.tw_violation == 0.0 and er.time_end <= inst.depot_close:
            current.append(cid)
        else:
            if len(current) > 0:
                routes.append(current)
                veh_types.append(veh_type)
            current = [cid]

    if len(current) > 0:
        routes.append(current)
        veh_types.append(veh_type)

    return routes, veh_types
