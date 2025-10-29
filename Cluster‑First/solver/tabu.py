from __future__ import annotations
from typing import List, Tuple, Dict
import random

from .data import Instance
from .evaluator import route_cost_with_penalties
from .neighborhoods import (
    two_opt_neighbors, apply_two_opt,
    relocate_neighbors, apply_relocate,
    swap_neighbors, apply_swap
)

class TabuSearch:
    def __init__(self, lamQ=1000.0, lamT=100.0, tenure_min=7, tenure_max=25,
                 max_iter=2000, no_improve_limit=200, rng_seed=None):
        self.lamQ = lamQ
        self.lamT = lamT
        self.tenure_min = tenure_min
        self.tenure_max = tenure_max
        self.max_iter = max_iter
        self.no_improve_limit = no_improve_limit
        self.rng = random.Random(rng_seed)

    def improve_route(self, inst: Instance, route: List[int], veh_type: int = 0) -> List[int]:
        """
        Tabu Search intra-route simple mêlant 2-opt, relocate et swap.
        """
        best = route[:]
        best_cost = route_cost_with_penalties(inst, best, veh_type, self.lamQ, self.lamT)
        cur = best[:]
        tabu: Dict[tuple, int] = {}
        cur_iter = 0
        since_best = 0

        while cur_iter < self.max_iter and since_best < self.no_improve_limit:
            cur_iter += 1
            since_best += 1
            move_best = None
            move_best_cost = float("inf")
            move_apply = None

            # On mélange l'ordre des mouvements pour diversité
            for move_type in self.rng.sample(["2opt", "reloc", "swap"], 3):
                if move_type == "2opt":
                    for (i, j) in two_opt_neighbors(cur):
                        cand = apply_two_opt(cur, i, j)
                        cost = route_cost_with_penalties(inst, cand, veh_type, self.lamQ, self.lamT)
                        key = ("2opt", i, j)
                        # Règle tabou + aspiration
                        if tabu.get(key, 0) > cur_iter and cost >= best_cost:
                            continue
                        if cost < move_best_cost:
                            move_best_cost = cost
                            move_best = cand
                            move_apply = key

                elif move_type == "reloc":
                    for (i, j) in relocate_neighbors(cur):
                        cand = apply_relocate(cur, i, j)
                        cost = route_cost_with_penalties(inst, cand, veh_type, self.lamQ, self.lamT)
                        key = ("reloc", i, j)
                        if tabu.get(key, 0) > cur_iter and cost >= best_cost:
                            continue
                        if cost < move_best_cost:
                            move_best_cost = cost
                            move_best = cand
                            move_apply = key

                else:  # swap
                    for (i, j) in swap_neighbors(cur):
                        cand = apply_swap(cur, i, j)
                        cost = route_cost_with_penalties(inst, cand, veh_type, self.lamQ, self.lamT)
                        key = ("swap", i, j)
                        if tabu.get(key, 0) > cur_iter and cost >= best_cost:
                            continue
                        if cost < move_best_cost:
                            move_best_cost = cost
                            move_best = cand
                            move_apply = key

            if move_best is None:
                break

            cur = move_best
            # Tenure dynamique
            tenure = self.rng.randint(self.tenure_min, self.tenure_max)
            tabu[move_apply] = cur_iter + tenure

            if move_best_cost < best_cost:
                best = cur[:]
                best_cost = move_best_cost
                since_best = 0

        return best
