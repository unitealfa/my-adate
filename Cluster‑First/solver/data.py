from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf, isfinite, ceil
from pathlib import Path
from typing import List
import re


@dataclass
class Instance:
    name: str
    n: int                           # nb clients
    coords: List[List[float]]        # shape (n+1, 2) avec dépôt en index 0
    demand: List[float]              # shape (n+1,), demand[0]=0
    service: List[float]             # shape (n+1,), service times
    window_a: List[float]            # shape (n+1,), earliest
    window_b: List[float]            # shape (n+1,), latest
    capacity: List[float]            # capacités des types de véhicules, shape (T,)
    num_veh_by_type: List[float]     # cardinalité par type (T,), ou +inf pour illimité
    cost_per_km: List[float]         # coût variable par type (T,)
    fixed_cost: List[float]          # coût fixe par type (T,)
    dist: List[List[float]]          # matrice distances (n+1, n+1)
    time: List[List[float]]          # matrice temps (n+1, n+1) ; = dist si pas d’info
    candidate_lists: List[List[int]] # voisins proches (indices)
    depot_open: float                # a0
    depot_close: float               # b0


def _distance_with_type(a: List[float], b: List[float], edge_type: str) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    euclid = hypot(dx, dy)
    etype = edge_type.upper()

    if etype == "EUC_2D":
        # Arrondi standard VRPLIB
        return float(int(euclid + 0.5))
    if etype == "CEIL_2D":
        return float(ceil(euclid))
    if etype == "MAN_2D":
        return float(abs(dx) + abs(dy))

    # Fallback: distance Euclidienne exacte
    return euclid


def build_candidate_lists(dist: List[List[float]], m: int = 20) -> List[List[int]]:
    n = len(dist)
    lists: List[List[int]] = []
    for i in range(n):
        order = sorted(range(n), key=lambda j: dist[i][j])
        order = [idx for idx in order if idx != i]
        lists.append(order[:m])
    return lists


def _finalize_instance(
    name: str,
    coords: List[List[float]],
    demand: List[float],
    service: List[float],
    window_a: List[float],
    window_b: List[float],
    capacity: float,
    vehicle_count: float | None,
    speed: float,
    candidate_m: int,
    edge_weight_type: str = "EUC_2D",
) -> Instance:
    dim = len(coords)
    dist: List[List[float]] = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            dist[i][j] = _distance_with_type(coords[i], coords[j], edge_weight_type)

    factor = 1.0 / max(1e-9, speed)
    time: List[List[float]] = [[dist[i][j] * factor for j in range(dim)] for i in range(dim)]
    clists = build_candidate_lists(dist, m=candidate_m)

    depot_open = window_a[0] if window_a[0] > 0 else 0.0
    depot_close = window_b[0] if isfinite(window_b[0]) else 1e9

    return Instance(
        name=name,
        n=dim - 1,
        coords=coords,
        demand=demand,
        service=service,
        window_a=window_a,
        window_b=window_b,
        capacity=[capacity],
        num_veh_by_type=[vehicle_count if vehicle_count is not None else 1e9],
        cost_per_km=[1.0],
        fixed_cost=[0.0],
        dist=dist,
        time=time,
        candidate_lists=clists,
        depot_open=depot_open,
        depot_close=depot_close,
    )


def _parse_cvrp(path: Path, speed: float, candidate_m: int) -> Instance:
    meta: dict[str, str] = {}
    coords_map: dict[int, List[float]] = {}
    demand_map: dict[int, float] = {}

    section: str | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("NODE_COORD_SECTION"):
                section = "NODE"
                continue
            if upper.startswith("DEMAND_SECTION"):
                section = "DEMAND"
                continue
            if upper.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            if upper.startswith("EOF"):
                break
            if ":" in line and section is None:
                key, value = line.split(":", 1)
                meta[key.strip().upper()] = value.strip()
                continue

            if section == "NODE":
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    coords_map[idx] = [float(parts[1]), float(parts[2])]
            elif section == "DEMAND":
                parts = line.split()
                if len(parts) >= 2:
                    demand_map[int(parts[0])] = float(parts[1])
            elif section == "DEPOT":
                # DEPOT_SECTION se termine par -1. Rien de particulier à stocker.
                if line.startswith("-"):
                    section = None

    dim = int(meta.get("DIMENSION", str(max(coords_map) if coords_map else 0)))
    coords: List[List[float]] = [[0.0, 0.0] for _ in range(dim)]
    for idx, xy in coords_map.items():
        if 1 <= idx <= dim:
            coords[idx - 1] = xy

    demand: List[float] = [0.0 for _ in range(dim)]
    for idx, value in demand_map.items():
        if 1 <= idx <= dim:
            demand[idx - 1] = value

    service = [0.0 for _ in range(dim)]
    window_a = [0.0 for _ in range(dim)]
    window_b = [inf for _ in range(dim)]

    capacity = float(meta.get("CAPACITY", "1e12"))
    vehicles = meta.get("VEHICLES")
    veh_count = float(vehicles) if vehicles is not None else None
    if veh_count is None:
        comment = meta.get("COMMENT")
        if comment:
            match = re.search(r"(no\s+of\s+trucks|number\s+of\s+vehicles)\s*[:=]\s*(\d+)", comment, re.IGNORECASE)
            if match:
                veh_count = float(match.group(2))
    edge_type = meta.get("EDGE_WEIGHT_TYPE", "EUC_2D")

    if veh_count is None:
        inferred_name = meta.get("NAME", path.name)
        match = re.search(r"-k(\d+)", inferred_name, re.IGNORECASE)
        if match:
            veh_count = float(match.group(1))

    name = meta.get("NAME", path.name)
    return _finalize_instance(name, coords, demand, service, window_a, window_b,
                              capacity, veh_count, speed, candidate_m, edge_type)


def _parse_solomon(path: Path, speed: float, candidate_m: int) -> Instance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    it = iter(lines)

    try:
        name = next(it)
    except StopIteration:  # pragma: no cover - fichier vide
        raise ValueError(f"Fichier Solomon vide: {path}")

    num_veh = None
    capacity = None
    for line in it:
        if line.upper().startswith("VEHICLE"):
            break
    for line in it:
        if line.upper().startswith("NUMBER"):
            break
    try:
        line = next(it)
        parts = line.split()
        if len(parts) >= 2:
            num_veh = float(parts[0])
            capacity = float(parts[1])
    except StopIteration:
        pass

    for line in it:
        if line.upper().startswith("CUSTOMER"):
            break
    # En-tête des colonnes
    try:
        header = next(it)
    except StopIteration:
        raise ValueError(f"Section CUSTOMER manquante dans {path}")
    header_parts = header.split()
    if len(header_parts) < 7:
        raise ValueError(f"Format CUSTOMER inattendu dans {path}")

    coords_map: dict[int, List[float]] = {}
    demand_map: dict[int, float] = {}
    service_map: dict[int, float] = {}
    window_a_map: dict[int, float] = {}
    window_b_map: dict[int, float] = {}

    for line in it:
        parts = line.split()
        if len(parts) < 7:
            continue
        idx = int(parts[0])
        coords_map[idx] = [float(parts[1]), float(parts[2])]
        demand_map[idx] = float(parts[3])
        window_a_map[idx] = float(parts[4])
        window_b_map[idx] = float(parts[5])
        service_map[idx] = float(parts[6])

    dim = max(coords_map) + 1 if coords_map else 0
    coords: List[List[float]] = [[0.0, 0.0] for _ in range(dim)]
    demand: List[float] = [0.0 for _ in range(dim)]
    service: List[float] = [0.0 for _ in range(dim)]
    window_a: List[float] = [0.0 for _ in range(dim)]
    window_b: List[float] = [inf for _ in range(dim)]

    for idx in range(dim):
        if idx in coords_map:
            coords[idx] = coords_map[idx]
        if idx in demand_map:
            demand[idx] = demand_map[idx]
        if idx in service_map:
            service[idx] = service_map[idx]
        if idx in window_a_map:
            window_a[idx] = window_a_map[idx]
        if idx in window_b_map:
            window_b[idx] = window_b_map[idx]

    if capacity is None:
        capacity = 1e12

    return _finalize_instance(name, coords, demand, service, window_a, window_b,
                              capacity, num_veh, speed, candidate_m, "EUC_2D")


def load_vrplib(path_or_name: str, speed: float = 1.0, candidate_m: int = 20) -> Instance:
    """
    Charge une instance VRPLIB/CVRPLIB/Solomon.
    - path_or_name: chemin local (recommandé) ou nom "X-n101-k25.vrp" présent dans ton dossier.
    - speed: si tu veux convertir distance -> temps (time = dist / speed).
    """
    path = Path(path_or_name)
    if not path.exists():
        guess = Path(__file__).resolve().parent.parent / "data" / path
        if guess.exists():
            path = guess
        else:
            raise FileNotFoundError(f"Instance introuvable: {path_or_name}")

    suffix = path.suffix.lower()
    sample = path.read_text(encoding="utf-8", errors="ignore")
    upper_sample = sample.upper()
    if suffix in {".vrp", ".vrptw", ".vrpl"} or "NODE_COORD_SECTION" in upper_sample:
        return _parse_cvrp(path, speed, candidate_m)
    if suffix in {".txt", ".solomon"} or "CUSTOMER" in upper_sample:
        return _parse_solomon(path, speed, candidate_m)

    raise ValueError(f"Format d'instance non reconnu pour {path}")
