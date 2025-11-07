from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .cost import CostEvaluator

@dataclass
class Solution:
    routes: List[List[int]]  # chaque route commence et finit par 0
    cost: float = field(default=float("inf"))
    dist: float = 0.0
    time_warp: float = 0.0
    last_return: float = 0.0

    def evaluate(self, evaluator: CostEvaluator):
        c, d, tw, lr = evaluator.solution_cost(self.routes)
        self.cost, self.dist, self.time_warp, self.last_return = c, d, tw, lr
        return self

    def is_feasible(self) -> bool:
        return self.time_warp == 0.0

    def clone(self) -> "Solution":
        return Solution(routes=[r[:] for r in self.routes], cost=self.cost, dist=self.dist,
                        time_warp=self.time_warp, last_return=self.last_return)
