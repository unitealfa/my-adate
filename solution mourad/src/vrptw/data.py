from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json, math
import numpy as np

@dataclass
class Depot:
    x: float
    y: float
    tw_early: float
    tw_late: float
    service: float = 0.0

@dataclass
class Client:
    id: int
    x: float
    y: float
    tw_early: float
    tw_late: float
    service: float

@dataclass
class ProblemData:
    depot: Depot
    clients: List[Client]
    dist: np.ndarray  # (n+1, n+1) 0 = dépôt
    dur: np.ndarray   # même dimensions

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    def coords(self) -> List[Tuple[float, float]]:
        return [(self.depot.x, self.depot.y)] + [(c.x, c.y) for c in self.clients]

def load_dataset(path: str) -> Tuple[Depot, List[Client]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    d = obj["depot"]
    depot = Depot(**d)
    clients = [Client(**c) for c in obj["clients"]]
    return depot, clients

def euclidean_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            M[i, j] = math.hypot(xi - xj, yi - yj)
    return M

def build_data(path: str) -> ProblemData:
    depot, clients = load_dataset(path)
    coords = [(depot.x, depot.y)] + [(c.x, c.y) for c in clients]
    dist = euclidean_matrix(coords)
    dur = dist.copy()  # durée = distance (simple). Remplace si tu as une vraie matrice de temps.
    return ProblemData(depot=depot, clients=clients, dist=dist, dur=dur)

def save_dataset(depot: Depot, clients: List[Client], path: str) -> None:
    obj: Dict[str, Any] = {"depot": depot.__dict__, "clients": [c.__dict__ for c in clients]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
