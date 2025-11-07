from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .data import ProblemData
from .schedule import forward_schedule

class CostTerms:
    def __init__(self, w_dist=1.0, w_tw=20.0, w_dur=5.0):
        self.w_dist = w_dist
        self.w_tw = w_tw
        self.w_dur = w_dur

class CostEvaluator:
    def __init__(self, data: ProblemData, penalties: CostTerms | None, shift_duration: float | None):
        self.data = data
        self.pen = penalties if penalties is not None else CostTerms()
        self.shift = shift_duration

    def route_distance(self, route: List[int]) -> float:
        D = 0.0
        for a, b in zip(route[:-1], route[1:]):
            D += float(self.data.dist[a, b])
        return D

    def route_cost(self, route: List[int]) -> tuple[float, float, float, float]:
        # retourne (distance, time_warp, overtime, duration)
        _, _, _, tw, dur = forward_schedule(route, self.data)
        dist = self.route_distance(route)
        overtime = max(0.0, dur - self.shift) if self.shift is not None else 0.0
        return dist, tw, overtime, dur

    def solution_cost(self, routes: List[List[int]]) -> tuple[float, float, float, float]:
        tot_dist = 0.0; tot_tw = 0.0; tot_ot = 0.0; last_return = 0.0
        for r in routes:
            dist, tw, ot, dur = self.route_cost(r)
            tot_dist += dist; tot_tw += tw; tot_ot += ot
            last_return = max(last_return, dur)
        penalized = self.pen.w_dist * tot_dist + self.pen.w_tw * tot_tw + self.pen.w_dur * tot_ot
        return penalized, tot_dist, tot_tw, last_return
