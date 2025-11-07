from __future__ import annotations
from .ls_ops import relocate, two_opt_star, swap_1_1
from .solution import Solution
from .cost import CostEvaluator

# Pipeline LS: appliquer des moves tant que amélioration
def improve(solution: Solution, evaluator: CostEvaluator, max_rounds=10) -> Solution:
    best = solution
    for _ in range(max_rounds):
        improved = False
        for op in (relocate, swap_1_1, two_opt_star):
            if op(best):
                best.evaluate(evaluator)
                improved = True
        if not improved:
            break
    return best
