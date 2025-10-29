from __future__ import annotations
from typing import List, Tuple, Iterable

def two_opt_neighbors(route: List[int]) -> Iterable[Tuple[int, int]]:
    """Génère des paires (i,j) pour inverser le segment [i:j] (2-opt)."""
    n = len(route)
    for i in range(n - 1):
        for j in range(i + 2, n + 1):
            yield (i, j)

def apply_two_opt(route: List[int], i: int, j: int) -> List[int]:
    return route[:i] + list(reversed(route[i:j])) + route[j:]

def relocate_neighbors(route: List[int]) -> Iterable[Tuple[int, int]]:
    """Déplacement d'un client de pos i vers pos j (i!=j)."""
    n = len(route)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            yield (i, j)

def apply_relocate(route: List[int], i: int, j: int) -> List[int]:
    r = route.copy()
    node = r.pop(i)
    r.insert(j, node)
    return r

def swap_neighbors(route: List[int]) -> Iterable[Tuple[int, int]]:
    n = len(route)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (i, j)

def apply_swap(route: List[int], i: int, j: int) -> List[int]:
    r = route.copy()
    r[i], r[j] = r[j], r[i]
    return r
