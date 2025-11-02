from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from .data import Instance

@dataclass
class RouteEval:
    feasible: bool
    cost: float
    distance: float
    time_end: float
    load: float
    waiting_total: float
    tw_violation: float
    capacity_excess: float


from __future__ import annotations
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import dataclass
from typing import List, Tuple
from typing import List, Tuple

from .data import Instance
from .data import Instance


@dataclass
@dataclass
class RouteEval:
class RouteEval:
    feasible: bool
    feasible: bool
    cost: float
    cost: float
    distance: float
    distance: float
    time_end: float
    time_end: float
    load: float
    load: float
    waiting_total: float
    waiting_total: float
    tw_violation: float
    tw_violation: float
    capacity_excess: float
    capacity_excess: float



@dataclass
class RouteState:
    """Structure de données permettant une évaluation incrémentale.

    On mémorise les valeurs cumulées le long de la route courante afin de
    recalculer rapidement le coût d'un voisin généré par Tabu Search.
    """

    inst: Instance
    veh_type: int
    lamQ: float
    lamT: float
    route: List[int]

    def __post_init__(self) -> None:
        self.update(self.route)

    def _build_prefix(self) -> None:
        seq = self.seq
        inst = self.inst

        n = len(seq)
        self.dist_prefix = [0.0] * n
        self.leave_time = [inst.depot_open] * n
        self.waiting_prefix = [0.0] * n
        self.tw_prefix = [0.0] * n
        self.load_after = [0.0] * n
        self.max_load_prefix = [0.0] * n

        time = inst.depot_open
        dist = 0.0
        waiting = 0.0
        tw = 0.0
        load = 0.0
        max_load = 0.0

        for idx in range(1, n):
            prev = seq[idx - 1]
            cur = seq[idx]

            dist += inst.dist[prev][cur]
            time += inst.time[prev][cur]

            if idx < n - 1:
                load += inst.demand[cur]
                if load > max_load:
                    max_load = load
                if time < inst.window_a[cur]:
                    waiting += inst.window_a[cur] - time
                    time = inst.window_a[cur]
                if time > inst.window_b[cur]:
                    tw += time - inst.window_b[cur]
                time += inst.service[cur]

            self.dist_prefix[idx] = dist
            self.leave_time[idx] = time
            self.waiting_prefix[idx] = waiting
            self.tw_prefix[idx] = tw
            self.load_after[idx] = load
            self.max_load_prefix[idx] = max_load

    def _prefix_state(self, idx: int) -> Tuple[float, float, float, float, float, float]:
        if idx < 0:
            return 0.0, self.inst.depot_open, 0.0, 0.0, 0.0, 0.0
        idx = min(idx, len(self.seq) - 1)
        return (
            self.dist_prefix[idx],
            self.leave_time[idx],
            self.waiting_prefix[idx],
            self.tw_prefix[idx],
            self.load_after[idx],
            self.max_load_prefix[idx],
        )

    def _evaluate_from(self, seq: List[int], start_idx: int) -> Tuple[float, float, float, float, float, float, float]:
        inst = self.inst
        veh = self.veh_type

        if start_idx <= 0:
            dist = 0.0
            time = inst.depot_open
            waiting = 0.0
            tw = 0.0
            load = 0.0
            max_load = 0.0
            start_range = 1
        else:
            dist, time, waiting, tw, load, max_load = self._prefix_state(start_idx - 1)
            start_range = start_idx

        for idx in range(start_range, len(seq)):
            prev = seq[idx - 1]
            cur = seq[idx]
            dist += inst.dist[prev][cur]
            time += inst.time[prev][cur]

            if idx < len(seq) - 1:
                node = cur
                load += inst.demand[node]
                if load > max_load:
                    max_load = load
                if time < inst.window_a[node]:
                    waiting += inst.window_a[node] - time
                    time = inst.window_a[node]
                if time > inst.window_b[node]:
                    tw += time - inst.window_b[node]
                time += inst.service[node]

        cap = inst.capacity[veh]
        capacity_excess = max(0.0, max_load - cap)
        used = 1 if len(seq) > 2 else 0
        cost = used * (inst.fixed_cost[veh] + inst.cost_per_km[veh] * dist)
        penalized = cost + self.lamQ * capacity_excess + self.lamT * tw
        return penalized, cost, dist, waiting, tw, capacity_excess, time

    def update(self, route: List[int]) -> float:
        self.route = route[:]
        self.seq = [0] + self.route + [0]
        self._build_prefix()
        (self.penalized_cost,
         self.raw_cost,
         self.distance,
         self.waiting,
         self.tw_violation,
         self.capacity_excess,
         self.time_end) = self._evaluate_from(self.seq, 0)
        return self.penalized_cost

    def evaluate_candidate(self, new_route: List[int]) -> float:
        seq = [0] + new_route + [0]
        if seq == self.seq:
            return self.penalized_cost
        limit = min(len(seq), len(self.seq))
        start_idx = 0
        while start_idx < limit and seq[start_idx] == self.seq[start_idx]:
            start_idx += 1
        return self._evaluate_from(seq, start_idx)[0]

def eval_route(instance: Instance, route: List[int], veh_type: int = 0) -> RouteEval:
    """
    route = liste d'indices de clients (sans le dépôt).
    On ajoute dépôt (0) au début et à la fin pour l’évaluation.
    """
    cap = instance.capacity[veh_type]
    ckm = instance.cost_per_km[veh_type]
    fcx = instance.fixed_cost[veh_type]

    seq = [0] + route + [0]
    t = instance.depot_open
    load = 0.0
    waiting = 0.0
    tw_violation = 0.0
    dist = 0.0

    # Distance totale
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        dist += instance.dist[a][b]

    # Propagation temporelle (forward pass)
    t = instance.depot_open
    for idx in range(1, len(seq)):
        i_prev = seq[idx - 1]
        i = seq[idx]
        travel = instance.time[i_prev][i]
        t = t + travel

        # Client (pas dépôt final)
        if idx < len(seq) - 1:
            load += instance.demand[i]
            # Attente si on arrive avant le début de fenêtre
            if t < instance.window_a[i]:
                waiting += (instance.window_a[i] - t)
                t = instance.window_a[i]
            # Violation si on dépasse la fin de fenêtre
            if t > instance.window_b[i]:
                tw_violation += (t - instance.window_b[i])
            # Service
            t += instance.service[i]

    capacity_excess = max(0.0, load - cap)
    is_used = 1 if len(route) > 0 else 0
    cost = is_used * (fcx + ckm * dist)
    feasible = (capacity_excess == 0.0) and (tw_violation == 0.0) and (t <= instance.depot_close)

    return RouteEval(
        feasible=feasible,
        cost=cost,
        distance=dist,
        time_end=t,
        load=load,
        waiting_total=waiting,
        tw_violation=tw_violation,
        capacity_excess=capacity_excess,
    )

def route_cost_with_penalties(inst: Instance, route: List[int], veh_type: int,
                              lamQ: float, lamT: float) -> float:
    r = eval_route(inst, route, veh_type)
    return r.cost + lamQ * r.capacity_excess + lamT * r.tw_violation

def solution_cost(inst: Instance, routes: List[List[int]], veh_types: List[int],
                  lamQ: float, lamT: float) -> tuple[float, float]:
    tot = 0.0
    tot_dist = 0.0
    for r, k in zip(routes, veh_types):
        er = eval_route(inst, r, k)
        tot += er.cost + lamQ * er.capacity_excess + lamT * er.tw_violation
        tot_dist += er.distance
    return tot, tot_dist
