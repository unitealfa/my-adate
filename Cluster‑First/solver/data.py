from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from math import hypot
import os

@dataclass
class Instance:
    name: str
    n: int                           # nb clients
    coords: np.ndarray               # shape (n+1, 2) avec dépôt en index 0
    demand: np.ndarray               # shape (n+1,), demand[0]=0
    service: np.ndarray              # shape (n+1,), service times
    window_a: np.ndarray             # shape (n+1,), earliest
    window_b: np.ndarray             # shape (n+1,), latest
    capacity: np.ndarray             # capacités des types de véhicules, shape (T,)
    num_veh_by_type: np.ndarray      # cardinalité par type (T,), ou np.array([+inf]) pour illimité
    cost_per_km: np.ndarray          # coût variable par type (T,)
    fixed_cost: np.ndarray           # coût fixe par type (T,)
    dist: np.ndarray                 # matrice distances (n+1, n+1)
    time: np.ndarray                 # matrice temps (n+1, n+1) ; = dist si pas d’info
    candidate_lists: List[np.ndarray]# voisins proches (indices)
    depot_open: float                # a0
    depot_close: float               # b0

def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return hypot(a[0]-b[0], a[1]-b[1])

def build_candidate_lists(dist: np.ndarray, m: int = 20) -> List[np.ndarray]:
    n = dist.shape[0]
    lists = []
    for i in range(n):
        order = np.argsort(dist[i])
        order = order[order != i]
        lists.append(order[:m])
    return lists

def load_vrplib(path_or_name: str, speed: float = 1.0, candidate_m: int = 20) -> Instance:
    """
    Charge une instance VRPLIB/CVRPLIB/Solomon.
    - path_or_name: chemin local (recommandé) ou nom "X-n101-k25.vrp" présent dans ton dossier.
    - speed: si tu veux convertir distance -> temps (time = dist / speed).
    """
    try:
        import vrplib  # type: ignore
    except Exception as e:
        raise RuntimeError("Le paquet 'vrplib' est requis (pip install vrplib).") from e

    # Si on a passé juste un nom, on le cherche à côté du script courant
    if not os.path.exists(path_or_name):
        # tente dans ../data
        guess = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", path_or_name))
        if os.path.exists(guess):
            path_or_name = guess

    raw = vrplib.read_instance(path_or_name)  # lit fichier local (.vrp/.txt/.vrptw, etc.)

    dim = int(raw.get("dimension"))
    coords = np.zeros((dim, 2), dtype=float)

    # node_coord peut être list de tuples ou dict selon la source
    nc = raw.get("node_coord")
    if isinstance(nc, list):
        for row in nc:
            # formats possibles: (i, x, y) ou [i, x, y]
            i, x, y = int(row[0]), float(row[1]), float(row[2])
            coords[i-1] = (x, y)
    elif isinstance(nc, dict):
        for i, (x, y) in nc.items():
            coords[int(i)-1] = (float(x), float(y))
    else:
        raise ValueError("Format 'node_coord' non reconnu dans le fichier.")

    # Demandes
    demand = np.zeros(dim, dtype=float)
    dem = raw.get("demand", [])
    if isinstance(dem, list):
        for row in dem:
            i, d = int(row[0]), float(row[1])
            demand[i-1] = d
    elif isinstance(dem, dict):
        for i, d in dem.items():
            demand[int(i)-1] = float(d)

    # Fenêtres temporelles (si présentes)
    window_a = np.zeros(dim, dtype=float)
    window_b = np.full(dim, np.inf, dtype=float)
    if "time_window" in raw:
        tw = raw["time_window"]
        if isinstance(tw, list):
            for row in tw:
                i, a, b = int(row[0]), float(row[1]), float(row[2])
                window_a[i-1] = a
                window_b[i-1] = b
        elif isinstance(tw, dict):
            for i, (a, b) in tw.items():
                window_a[int(i)-1] = float(a)
                window_b[int(i)-1] = float(b)

    # Durées de service (si présentes)
    service = np.zeros(dim, dtype=float)
    if "service_time" in raw:
        st = raw["service_time"]
        if isinstance(st, list):
            for row in st:
                i, s = int(row[0]), float(row[1])
                service[i-1] = s
        elif isinstance(st, dict):
            for i, s in st.items():
                service[int(i)-1] = float(s)

    # Horizon dépôt
    depot_open  = window_a[0] if window_a[0] > 0 else 0.0
    depot_close = window_b[0] if np.isfinite(window_b[0]) else 1e9

    # Matrices dist/time (euclidiennes)
    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = _euclid(coords[i], coords[j])

    time = dist / max(1e-9, speed)

    # Flotte homogène par défaut (1 type)
    cap = np.array([float(raw.get("capacity", 1e12))], dtype=float)
    num = np.array([1e9], dtype=float)        # illimité pour commencer
    ckm = np.array([1.0], dtype=float)
    fcx = np.array([0.0], dtype=float)

    # Candidate lists
    clists = build_candidate_lists(dist, m=candidate_m)

    return Instance(
        name=os.path.basename(path_or_name),
        n=n-1,
        coords=coords,
        demand=demand,
        service=service,
        window_a=window_a,
        window_b=window_b,
        capacity=cap,
        num_veh_by_type=num,
        cost_per_km=ckm,
        fixed_cost=fcx,
        dist=dist,
        time=time,
        candidate_lists=clists,
        depot_open=depot_open,
        depot_close=depot_close,
    )
