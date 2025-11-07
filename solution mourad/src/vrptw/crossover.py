from __future__ import annotations
from typing import List, Set
import random
from .solution import Solution

# SREX simplifié: garder un sous-ensemble de routes du parent A, compléter avec B, puis réparer
def srex(parentA: Solution, parentB: Solution, n_nodes: int) -> Solution:
    child_routes: List[List[int]] = []

    # 1) copier q routes de A
    routesA = parentA.routes[:]
    random.shuffle(routesA)
    q = max(1, len(routesA)//2)
    kept: Set[int] = set()
    for r in routesA[:q]:
        child_routes.append(r[:])
        kept.update(r[1:-1])

    # 2) compléter avec routes de B si elles apportent des clients non encore couverts
    for r in parentB.routes:
        add = [v for v in r[1:-1] if v not in kept]
        if not add:
            continue
        child_routes.append([0] + add + [0])
        kept.update(add)

    # 3) réparer: insérer les clients manquants
    missing = [v for v in range(1, n_nodes) if v not in kept]
    for v in missing:
        # insertion naïve dans la route la plus courte
        best_r, best_pos, best_incr = None, None, float("inf")
        for r in child_routes:
            for pos in range(1, len(r)):
                incr = 1.0  # TODO: remplacer par un vrai delta de distance
                if incr < best_incr:
                    best_r, best_pos, best_incr = r, pos, incr
        if best_r is None:
            child_routes.append([0, v, 0])
        else:
            best_r.insert(best_pos, v)

    return Solution(routes=child_routes)
