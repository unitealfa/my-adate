from __future__ import annotations
import random, time
from typing import List
from .solution import Solution
from .population import Population
from .penalty import PenaltyManager
from .cost import CostEvaluator, CostTerms
from .local_search import improve
from .crossover import srex

class GAParams:
    def __init__(self, time_limit_s=60, init_size=25):
        self.time_limit_s = time_limit_s
        self.init_size = init_size

def make_random_solution(n_nodes: int, k: int) -> Solution:
    clients = list(range(1, n_nodes))
    random.shuffle(clients)
    routes = [[] for _ in range(k)]
    # répartition simple round-robin
    for i, v in enumerate(clients):
        routes[i % k].append(v)
    routes = [[0] + r + [0] for r in routes if r]
    if not routes:
        routes = [[0, 0]]
    return Solution(routes=routes)

def run_hgs(n_nodes: int, k: int, evaluator: CostEvaluator, params: GAParams) -> Solution:
    pop = Population()
    pen = PenaltyManager()

    # init population
    for _ in range(params.init_size):
        s = make_random_solution(n_nodes, k).evaluate(evaluator)
        pop.add(s)
        pen.register(s.is_feasible())

    best = min(pop.individuals, key=lambda s: s.cost)
    t_end = time.time() + params.time_limit_s

    while time.time() < t_end:
        A, B = pop.select_parents()
        child = srex(A, B, n_nodes)
        child = improve(child, evaluator)
        child.evaluate(evaluator)
        pop.add(child)
        pen.register(child.is_feasible())

        # (optionnel) ajuster les poids de pénalité dans l'évaluateur selon le PenaltyManager
        evaluator.pen = pen.cost_terms()

        if len(pop.individuals) >= pop.min_size + pop.gen_size:
            pop.survivor_selection()
        if child.cost < best.cost:
            best = child

    return best
