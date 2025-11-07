from __future__ import annotations
from typing import List, Tuple
from .solution import Solution

# Opérateurs LS basiques (à compléter): relocate intra/inter, swap(1,1), 2-opt* inter-route

def relocate(sol: Solution) -> bool:
    # TODO: delta-évaluation réelle; pour l’instant, noop
    return False

def two_opt_star(sol: Solution) -> bool:
    # TODO: implémenter 2-opt* inter-route
    return False

def swap_1_1(sol: Solution) -> bool:
    # TODO: échange simple de deux clients de deux routes
    return False
