from __future__ import annotations
from .data import build_data, ProblemData
from .cost import CostEvaluator, CostTerms
from .ga import run_hgs, GAParams
from .viz import plot_routes

class SolveParams:
    def __init__(self, k: int, shift_duration: float | None, time_limit_s: int):
        self.k = k
        self.shift = shift_duration
        self.time_limit_s = time_limit_s

def solve(path_json: str, k: int, shift_duration: float | None, time_limit_s: int):
    data: ProblemData = build_data(path_json)
    eval = CostEvaluator(data, penalties=CostTerms(), shift_duration=shift_duration)
    params = GAParams(time_limit_s=time_limit_s)
    best = run_hgs(n_nodes=data.dist.shape[0], k=k, evaluator=eval, params=params)
    return data, best
