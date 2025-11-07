from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .data import ProblemData

# Forward schedule sur une route: calcule arrivées, départs, time_warp (retard), durée, etc.
# Route = liste d’indices noeuds dans l’indexation globale: 0 = dépôt, 1..n = clients.

def forward_schedule(route: List[int], data: ProblemData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    tw_early = [data.depot.tw_early] + [c.tw_early for c in data.clients]
    tw_late  = [data.depot.tw_late ] + [c.tw_late  for c in data.clients]
    service  = [data.depot.service ] + [c.service  for c in data.clients]

    nstops = len(route)
    arr = np.zeros(nstops)
    start = np.zeros(nstops)
    dep = np.zeros(nstops)

    time_warp = 0.0

    # départ dépôt (noeud 0 de la route)
    arr[0] = max(0.0, tw_early[route[0]])
    start[0] = max(arr[0], tw_early[route[0]])
    dep[0] = start[0] + service[route[0]]

    for k in range(1, nstops):
        i, j = route[k-1], route[k]
        t = dep[k-1] + data.dur[i, j]
        # attente si trop tôt
        start[k] = max(t, tw_early[j])
        arr[k] = t
        # retard (TW dure): si start > tw_late => time_warp accumulé
        if start[k] > tw_late[j]:
            time_warp += start[k] - tw_late[j]
        dep[k] = start[k] + service[j]

    duration = dep[-1] - dep[0]
    last_return = dep[-1]
    return arr, start, dep, time_warp, duration
