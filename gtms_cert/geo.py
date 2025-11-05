"""Geometric and cost related helpers."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

try:  # Optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback when NumPy is unavailable
    np = None


EARTH_RADIUS_KM = 6371.0


def haversine(coord_a: Sequence[float], coord_b: Sequence[float]) -> float:
    """Compute the great-circle distance in kilometres."""
    lat1, lon1 = map(math.radians, coord_a)
    lat2, lon2 = map(math.radians, coord_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


@dataclass
class DistanceOracle:
    """Distance queries backed either by coordinates or a time matrix."""

    node_ids: List[str]
    depot_id: str
    metric: str
    avg_speed_kmh: float | None
    coordinates: Dict[str, Sequence[float]] | None
    matrix: Any | None
    symmetric: bool
    candidate_edges: Dict[str, List[tuple[str, float]]]

    def __post_init__(self) -> None:
        self._index: Dict[str, int] = {node: idx for idx, node in enumerate(self.node_ids)}
        self._cache: Dict[tuple[str, str], float] = {}

    def __getitem__(self, node: str) -> "DistanceRow":
        return DistanceRow(self, node)

    def nodes(self) -> List[str]:
        return list(self.node_ids)

    def get(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b)
        if key in self._cache:
            return self._cache[key]
        if self.matrix is not None:
            ia, ib = self._index[a], self._index[b]
            if np is not None and isinstance(self.matrix, np.ndarray):
                value = float(self.matrix[ia, ib])
            else:
                value = float(self.matrix[ia][ib])
        else:
            assert self.coordinates is not None and self.avg_speed_kmh is not None
            coord_a = self.coordinates[a]
            coord_b = self.coordinates[b]
            distance_km = haversine(coord_a, coord_b)
            value = (distance_km / (self.avg_speed_kmh / 60.0))
        if self.symmetric:
            self._cache[(b, a)] = value
        self._cache[key] = value
        return value

    def neighbors(self, node: str) -> List[tuple[str, float]]:
        return self.candidate_edges.get(node, [])

    def ensure_symmetric_candidates(self) -> None:
        for u, neighs in list(self.candidate_edges.items()):
            for v, cost in neighs:
                if all(w != u for w, _ in self.candidate_edges.get(v, [])):
                    self.candidate_edges.setdefault(v, []).append((u, cost))

    def route_time(self, sequence: Sequence[str]) -> float:
        total = 0.0
        for i in range(len(sequence) - 1):
            total += self.get(sequence[i], sequence[i + 1])
        return total


class DistanceRow(Mapping[str, float]):
    """Row view on the distance oracle."""

    def __init__(self, oracle: DistanceOracle, node: str) -> None:
        self.oracle = oracle
        self.node = node

    def __getitem__(self, key: str) -> float:  # type: ignore[override]
        return self.oracle.get(self.node, key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.oracle.node_ids)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.oracle.node_ids)


def build_distance_oracle(payload: Mapping[str, object], cands: int = 32) -> DistanceOracle:
    """Create a distance oracle from the JSON payload."""
    symmetric = bool(payload.get("symmetric", True))
    if "time_matrix_min" in payload:
        raw_matrix = payload["time_matrix_min"]
        if np is not None:
            matrix = np.array(raw_matrix, dtype=np.float32)
        else:
            matrix = [[float(val) for val in row] for row in raw_matrix]
        node_ids = list(payload.get("node_ids", []))
        depot_id = str(payload.get("depot_id"))
        metric = "matrix"
        coordinates = None
        avg_speed = None
    else:
        depot = payload["depot"]
        depot_id = depot["id"]
        customers = payload.get("customers", [])
        metric_info = payload.get("metric", {"type": "haversine", "avg_speed_kmh": 40})
        metric = metric_info.get("type", "haversine")
        avg_speed = float(metric_info.get("avg_speed_kmh", 40))
        node_ids = [depot_id] + [c["id"] for c in customers]
        coordinates = {
            depot_id: tuple(depot["coord"]),
            **{c["id"]: tuple(c["coord"]) for c in customers},
        }
        matrix = None
    oracle = DistanceOracle(
        node_ids=node_ids,
        depot_id=depot_id,
        metric=metric,
        avg_speed_kmh=avg_speed,
        coordinates=coordinates,
        matrix=matrix,
        symmetric=symmetric,
        candidate_edges=defaultdict(list),
    )
    _populate_candidates(oracle, cands=cands)
    oracle.ensure_symmetric_candidates()
    return oracle


def _populate_candidates(oracle: DistanceOracle, cands: int = 32) -> None:
    """Populate adjacency lists using k-nearest neighbours."""
    nodes = oracle.node_ids
    depot_id = oracle.depot_id
    if oracle.matrix is not None:
        mat = oracle.matrix
        size = len(nodes)
        k = min(cands, size - 1)
        for idx, node in enumerate(nodes):
            if node == depot_id:
                continue
            if np is not None and isinstance(mat, np.ndarray):
                row = mat[idx].astype(float)
                row[idx] = float("inf")
                nearest_idx = np.argpartition(row, k)[:k]
                oracle.candidate_edges[node] = [
                    (nodes[j], float(mat[idx, j])) for j in nearest_idx if nodes[j] != node
                ]
            else:
                row = mat[idx]
                candidates = [
                    (row[j], nodes[j]) for j in range(size) if j != idx
                ]
                candidates.sort(key=lambda x: x[0])
                oracle.candidate_edges[node] = [
                    (other, float(cost)) for cost, other in candidates[:k]
                ]
        depot_idx = oracle._index[depot_id]
        depot_edges = []
        if np is not None and isinstance(mat, np.ndarray):
            depot_edges = [
                (nodes[j], float(mat[depot_idx, j])) for j in range(size) if j != depot_idx
            ]
        else:
            depot_edges = [
                (nodes[j], float(mat[depot_idx][j])) for j in range(size) if j != depot_idx
            ]
        oracle.candidate_edges[depot_id] = depot_edges
        return

    assert oracle.coordinates is not None
    n = len(nodes)
    k = min(cands, n - 1)
    for node in nodes:
        if node == depot_id:
            continue
        candidates: List[tuple[float, str]] = []
        for other in nodes:
            if other == node:
                continue
            cost = oracle.get(node, other)
            candidates.append((cost, other))
        candidates.sort(key=lambda x: x[0])
        oracle.candidate_edges[node].extend((other, cost) for cost, other in candidates[:k])
    oracle.candidate_edges[depot_id] = [
        (other, oracle.get(depot_id, other)) for other in nodes if other != depot_id
    ]
