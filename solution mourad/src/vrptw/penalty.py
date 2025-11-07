from __future__ import annotations
from dataclasses import dataclass
from .cost import CostTerms

@dataclass
class PenaltyManager:
    target_feasible: float = 0.43
    increase: float = 1.3
    decrease: float = 0.7
    window: int = 50

    def __post_init__(self):
        self.pen = CostTerms(w_dist=1.0, w_tw=20.0, w_dur=5.0)
        self.recent = []  # 1 si faisable, 0 sinon

    def cost_terms(self) -> CostTerms:
        return self.pen

    def register(self, feasible: bool):
        self.recent.append(1 if feasible else 0)
        if len(self.recent) >= self.window:
            rate = sum(self.recent[-self.window:]) / self.window
            if rate < self.target_feasible:
                self.pen.w_tw *= self.increase
                self.pen.w_dur *= self.increase
            elif rate > self.target_feasible:
                self.pen.w_tw *= self.decrease
                self.pen.w_dur *= self.decrease
