from __future__ import annotations
from typing import List, Set, Tuple

def edges_of(routes: List[List[int]]) -> Set[Tuple[int,int]]:
    E = set()
    for r in routes:
        for a,b in zip(r[:-1], r[1:]):
            E.add((a,b))
    return E

# Broken Pairs Distance (0..1) entre deux solutions
def bpd(solA, solB) -> float:
    EA = edges_of(solA.routes)
    EB = edges_of(solB.routes)
    if not EA and not EB:
        return 0.0
    diff = len(EA.symmetric_difference(EB))
    denom = max(1, len(EA) + len(EB))
    return diff / denom
