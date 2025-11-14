#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRP / VRPTW solver (HGS + Split(TW) + RVND + ALNS, Numba-accelerated when available)

- Lit CVRPLIB .vrp (sans TW) et Solomon .txt (avec TW)
- Ne JAMAIS utiliser .sol pour construire : .sol uniquement pour le gap
- Distances: CVRP -> EUC_2D arrondies ; Solomon -> Euclidiennes
- Menu interactif : choix séparé des fichiers Avec TW (.txt) et Sans TW (.vrp)
- Auto-tuning des paramètres (-I, -P, -S, --fast, --nnk, --init, -W, -T) par instance
- Accélération : parallélisation CPU (joblib) + memmap + réduction sur-souscription BLAS
- Analyse rapide : --analyze <dir> [--match <substring>] pour recommander workers/fast/nnk
"""

# ------------------------------------------------------------
# (1) LOCK threads BLAS AVANT d'importer NumPy !
# ------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

# ------------------------------------------------------------
# Imports standard
# ------------------------------------------------------------
import math, re, sys, argparse, random, time, hashlib, threading, platform
from itertools import cycle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from typing import Union, Tuple
# ------------------------------------------------------------
# NumPy (après lock BLAS) + threadpoolctl (optionnel)
# ------------------------------------------------------------
import numpy as np
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)  # force MKL/BLAS à 1 thread par process
except Exception:
    pass

# ------------------------------------------------------------
# joblib (parallèle) avec memmap
# ------------------------------------------------------------
try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False
    def Parallel(*args, **kwargs):
        raise RuntimeError("joblib non disponible")
    def delayed(f):
        return f

# ------------------------------------------------------------
# numba (optionnel)
# ------------------------------------------------------------
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco

# ------------------------------ Data Model ------------------------------
@dataclass
class Instance:
    name: str
    n: int
    coords: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    capacity: int
    k: int
    dist: np.ndarray
    has_tw: bool

# ------------------------------ NN (granular) ------------------------------
def build_nn(dist: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Version rapide : utilise argpartition (O(n*k)) au lieu d'argsort complet (O(n log n)).
    On coupe à k+3 puis on trie localement ces candidats.
    """
    n = dist.shape[0]
    kth = min(k + 3, max(1, n - 1))
    part = np.argpartition(dist, kth=kth, axis=1)[:, :kth]
    nn_list: list[np.ndarray] = []
    for u in range(n):
        cand = part[u]
        cand = [int(v) for v in cand if v != u and v != 0]
        cand = sorted(cand, key=lambda v: dist[u, v])[:k]
        nn_list.append(np.asarray(cand, dtype=np.int32))
    return nn_list

# ------------------------------ Parsing ------------------------------
def parse_solomon_txt(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]
    lines = [ln.strip() for ln in lines if ln.strip()]
    num = cap = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith('VEHICLE'):
            for j in range(i+1, min(i+8, len(lines))):
                m = re.findall(r"(\d+)\s+(\d+)", lines[j])
                if m:
                    num, cap = map(int, m[0]); break
            break
    cust_start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith('CUSTOMER'):
            cust_start = i+1; break
    if cust_start is None:
        raise ValueError('CUSTOMER section not found in Solomon file')

    data = []
    for ln in lines[cust_start:]:
        if re.match(r"^\d+", ln):
            parts = re.split(r"\s+", ln.strip())
            if len(parts) >= 7:
                cid, x, y, dem, ready, due, service = parts[:7]
                data.append((int(cid), float(x), float(y), float(dem), float(ready), float(due), float(service)))
    data.sort(key=lambda t: t[0])
    n = len(data)-1
    if n <= 0:
        raise ValueError('No customers parsed from Solomon file')

    coords = np.zeros((n+1,2), dtype=np.float64)
    demand = np.zeros(n+1, dtype=np.float64)
    ready  = np.zeros(n+1, dtype=np.float64)
    due    = np.zeros(n+1, dtype=np.float64)
    service= np.zeros(n+1, dtype=np.float64)
    for cid, x, y, dem, r, d, s in data:
        coords[cid] = (x,y); demand[cid] = dem; ready[cid] = r; due[cid] = d; service[cid] = s

    # Euclidienne (pas d'arrondi)
    dist = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
    k = int(num) if num else n
    cap = int(cap) if cap else 10**9
    return Instance(Path(path).name, n, coords, demand, ready, due, service, cap, k, dist, True)

def parse_cvrplib_vrp(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    U     = [ln.upper() for ln in lines]

    def get_val(prefix: str) -> Optional[str]:
        for ln in lines:
            if ln.upper().startswith(prefix):
                parts = ln.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return None

    cap_str = get_val('CAPACITY') or '0'
    dim_str = get_val('DIMENSION') or None
    capacity = int(re.sub(r'\D','', cap_str)) if re.search(r'\d', cap_str or '') else 0
    dimension = int(re.sub(r'\D','', dim_str)) if re.search(r'\d', dim_str or '') else None

    def idx_of(section: str) -> Optional[int]:
        for i, ln in enumerate(U):
            if ln.startswith(section):
                return i
        return None

    node_idx = idx_of('NODE_COORD_SECTION')
    dem_idx  = idx_of('DEMAND_SECTION')
    dep_idx  = idx_of('DEPOT_SECTION')
    if node_idx is None or dem_idx is None or dep_idx is None:
        raise ValueError('Missing required sections in .vrp (NODE_COORD_SECTION/DEMAND_SECTION/DEPOT_SECTION).')

    coords_map: Dict[int, Tuple[float,float]] = {}
    i = node_idx + 1
    while i < len(lines) and not U[i].endswith('SECTION') and not U[i].startswith('DEPOT_SECTION') and not U[i].startswith('DEMAND_SECTION'):
        parts = re.split(r"\s+", lines[i])
        if len(parts) >= 3 and parts[0].isdigit():
            idx = int(parts[0]); x = float(parts[1]); y = float(parts[2]); coords_map[idx] = (x,y)
        i += 1

    demand_map: Dict[int, float] = {}
    i = dem_idx + 1
    while i < len(lines) and not U[i].startswith('DEPOT_SECTION') and not U[i].endswith('SECTION'):
        parts = re.split(r"\s+", lines[i])
        if len(parts) >= 2 and parts[0].isdigit():
            idx = int(parts[0]); demand_map[idx] = float(parts[1])
        i += 1

    depot_id = 1
    i = dep_idx + 1
    while i < len(lines):
        t = lines[i]
        if t == '-1': break
        if t.isdigit(): depot_id = int(t); break
        i += 1

    max_id = 0
    if coords_map: max_id = max(max_id, max(coords_map.keys()))
    if demand_map: max_id = max(max_id, max(demand_map.keys()))
    if dimension is None: dimension = max_id

    all_ids = [i for i in range(1, dimension+1) if i in coords_map]
    if depot_id not in coords_map: depot_id = 1
    cust_ids = [i for i in all_ids if i != depot_id]
    n = len(cust_ids)
    if n <= 0: raise ValueError('No customers parsed from .vrp file')

    coords2 = np.zeros((n+1,2), dtype=np.float64)
    demand2 = np.zeros(n+1, dtype=np.float64)
    ready   = np.zeros(n+1, dtype=np.float64)
    due     = np.full(n+1, 1e12, dtype=np.float64)
    service = np.zeros(n+1, dtype=np.float64)

    coords2[0] = coords_map[depot_id]
    for new_idx, orig in enumerate(cust_ids, start=1):
        coords2[new_idx] = coords_map.get(orig, (0.0,0.0))
        demand2[new_idx] = demand_map.get(orig, 0.0)

    # EUC_2D arrondies (convention CVRPLIB)
    diff = coords2[:, None, :] - coords2[None, :, :]
    dist = np.rint(np.sqrt((diff**2).sum(-1))).astype(np.float64)

    fname = Path(path).name
    m = re.search(r'-K(\d+)', fname, flags=re.IGNORECASE)
    if m: k = int(m.group(1))
    else:
        k = math.ceil(demand2[1:].sum()/capacity) if capacity>0 else n

    cap = int(capacity) if capacity>0 else 10**9
    return Instance(fname, n, coords2, demand2, ready, due, service, cap, int(k), dist, False)

# ------------------------------ Core Utils ------------------------------
@njit(cache=True, fastmath=True, nogil=True)
def route_cost(dist: np.ndarray, route: np.ndarray) -> float:
    c = 0.0
    for i in range(route.shape[0]-1):
        c += dist[route[i], route[i+1]]
    return c

def total_cost(inst: Instance, routes: List[List[int]]) -> float:
    return sum(route_cost(inst.dist, np.array(r, dtype=np.int32)) for r in routes)

class ConsoleLoader:
    def __init__(self, message: str = "Calcul en cours", delay: float = 0.25) -> None:
        self._message = message
        self._delay = delay
        self._symbols = cycle("|/-\\")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._enabled = sys.stdout.isatty()

    def _spin(self) -> None:
        while not self._stop.is_set():
            symbol = next(self._symbols)
            sys.stdout.write(f"\r{self._message} {symbol}")
            sys.stdout.flush()
            if self._stop.wait(self._delay):
                break
        sys.stdout.write("\r" + " " * (len(self._message) + 2) + "\r")
        sys.stdout.flush()

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        self._thread.join()

def capacity_ok(inst: Instance, route: List[int]) -> bool:
    load = 0.0
    for c in route[1:-1]:
        load += inst.demand[c]
        if load > inst.capacity + 1e-9:
            return False
    return True

def recompute_arrivals(inst: Instance, route: List[int]) -> Tuple[bool, List[float]]:
    arr = [0.0]
    for idx in range(1, len(route)):
        prev = route[idx-1]; cur = route[idx]
        depart_prev = max(arr[idx-1], inst.ready[prev]) + inst.service[prev]
        arr_cur = depart_prev + inst.dist[prev, cur]
        if arr_cur < inst.ready[cur]:
            arr_cur = inst.ready[cur]
        if arr_cur > inst.due[cur] + 1e-9:
            return False, []
        arr.append(arr_cur)
    return True, arr

# ------------------------------ Split (DP) ------------------------------
def split_cvrp(inst: Instance, tour: List[int]) -> Optional[List[List[int]]]:
    n = len(tour)
    cum_dem = np.zeros(n+1)
    for i in range(n):
        cum_dem[i+1] = cum_dem[i] + inst.demand[tour[i]]

    def route_cost_ij(i, j) -> float:
        load = cum_dem[j+1] - cum_dem[i]
        if load > inst.capacity + 1e-9: return 1e18
        c = inst.dist[0, tour[i]]
        for t in range(i, j):
            c += inst.dist[tour[t], tour[t+1]]
        c += inst.dist[tour[j], 0]
        return c

    INF = 1e18
    dp = [INF]*(n+1); prev = [-1]*(n+1); cnt = [10**9]*(n+1)
    dp[0] = 0.0; cnt[0] = 0
    for j in range(1, n+1):
        best = INF; bi = -1; bc = 10**9
        for i in range(j):
            c = route_cost_ij(i, j-1)
            if c >= INF: continue
            if cnt[i]+1 <= inst.k and dp[i]+c < best:
                best = dp[i]+c; bi = i; bc = cnt[i]+1
        dp[j] = best; prev[j] = bi; cnt[j] = bc
    if prev[n] == -1: return None
    routes = []
    cur = n
    while cur > 0:
        i = prev[cur]
        routes.append([0]+tour[i:cur]+[0])
        cur = i
    routes.reverse()
    return routes

def split_vrptw(inst: Instance, tour: List[int]) -> Optional[List[List[int]]]:
    n = len(tour)
    feas_cost = [[(False, 1e18)]*n for _ in range(n)]
    for i in range(n):
        load = 0.0
        r = [0]; arr = [0.0]
        ok = True
        for j in range(i, n):
            u = tour[j]
            load += inst.demand[u]
            if load > inst.capacity: ok = False
            prev = r[-1]
            depart_prev = max(arr[-1], inst.ready[prev]) + inst.service[prev]
            arr_u = depart_prev + inst.dist[prev, u]
            if arr_u < inst.ready[u]: arr_u = inst.ready[u]
            if arr_u > inst.due[u] + 1e-9: ok = False
            r.append(u); arr.append(arr_u)
            if ok:
                depart_u = max(arr[-1], inst.ready[u]) + inst.service[u]
                arr_dep = depart_u + inst.dist[u, 0]
                if arr_dep > inst.due[0] + 1e-9: ok = False
            if ok:
                cost = route_cost(inst.dist, np.array(r+[0], dtype=np.int32))
                feas_cost[i][j] = (True, cost)

    INF = 1e18
    dp = [INF]*(n+1); prev = [-1]*(n+1); cnt = [10**9]*(n+1)
    dp[0] = 0.0; cnt[0] = 0
    for j in range(1, n+1):
        best = INF; bi = -1; bc = 10**9
        for i in range(j):
            feas, c = feas_cost[i][j-1]
            if not feas: continue
            if cnt[i]+1 <= inst.k and dp[i]+c < best:
                best = dp[i]+c; bi = i; bc = cnt[i]+1
        dp[j] = best; prev[j] = bi; cnt[j] = bc
    if prev[n] == -1: return None
    routes = []
    cur = n
    while cur > 0:
        i = prev[cur]
        routes.append([0]+tour[i:cur]+[0])
        cur = i
    routes.reverse()
    return routes

# ------------------------------ Construction ------------------------------
def feasible_insert(inst: Instance, route: List[int], pos: int, u: int, arr_times: List[float]) -> Tuple[bool, float, List[float]]:
    load = sum(inst.demand[c] for c in route[1:-1]) + inst.demand[u]
    if load > inst.capacity + 1e-9: return False, 1e18, []
    i = route[pos-1]; j = route[pos]
    delta = inst.dist[i,u] + inst.dist[u,j] - inst.dist[i,j]
    new_arr = arr_times[:pos]
    new_route = route[:pos] + [u] + route[pos:]
    for idx in range(pos, len(new_route)):
        prev = new_route[idx-1]; cur = new_route[idx]
        depart_prev = max(new_arr[idx-1], inst.ready[prev]) + inst.service[prev]
        arr_cur = depart_prev + inst.dist[prev, cur]
        if arr_cur < inst.ready[cur]: arr_cur = inst.ready[cur]
        if arr_cur > inst.due[cur] + 1e-9: return False, 1e18, []
        new_arr.append(arr_cur)
    return True, delta, new_arr

def build_seed(inst: Instance) -> List[List[int]]:
    unserved = set(range(1, inst.n+1))
    routes: List[List[int]] = []
    arrivals: List[List[float]] = []
    while unserved:
        best = []
        for r_idx, r in enumerate(routes):
            arr = arrivals[r_idx]
            for u in list(unserved):
                b1 = (1e18, None, None); b2 = (1e18, None, None)
                for pos in range(1, len(r)):
                    feas, delta, new_arr = feasible_insert(inst, r, pos, u, arr)
                    if not feas: continue
                    if delta < b1[0]: b2 = b1; b1 = (delta, pos, new_arr)
                    elif delta < b2[0]: b2 = (delta, pos, new_arr)
                if b1[1] is not None:
                    regret = b2[0]-b1[0] if np.isfinite(b2[0]) else b1[0]
                    best.append((u, r_idx, b1[1], b1[0], b1[2], regret))
        if best:
            best.sort(key=lambda t: (-t[5], t[3]))
            u, r_idx, pos, _, new_arr, _ = best[0]
            routes[r_idx] = routes[r_idx][:pos] + [u] + routes[r_idx][pos:]
            arrivals[r_idx] = new_arr
            unserved.remove(u)
        else:
            seed = min(unserved, key=lambda u: inst.ready[u]) if inst.has_tw else \
                   max(unserved, key=lambda u: (inst.demand[u], inst.dist[0,u]))
            r = [0, seed, 0]
            ok, arr = recompute_arrivals(inst, r)
            if not ok:
                arr = [0.0, max(inst.dist[0,seed], inst.ready[seed]), 0.0]
            routes.append(r); arrivals.append(arr); unserved.remove(seed)
    return routes

def build_seed_sweep(inst: Instance) -> List[List[int]]:
    depot = inst.coords[0]
    vec = inst.coords[1:] - depot
    ang = np.arctan2(vec[:,1], vec[:,0])
    order = np.argsort(ang) + 1
    routes: List[List[int]] = []
    cur: List[int] = [0]; load = 0.0
    for u in order:
        d = float(inst.demand[int(u)])
        if load + d <= inst.capacity + 1e-9:
            cur.append(int(u)); load += d
        else:
            cur.append(0); routes.append(cur)
            cur = [0, int(u)]; load = d
    cur.append(0); routes.append(cur)
    return routes

# ------------------------------ RVND (mouvements optimisés) ------------------------------
def recompute_arrivals_safe(inst: Instance, r: List[int]) -> bool:
    ok, _ = recompute_arrivals(inst, r)
    return ok

def or_opt_intra(inst: Instance, r: List[int]) -> Tuple[bool, List[int]]:
    if len(r) <= 4: return False, r
    base = route_cost(inst.dist, np.array(r, dtype=np.int32))
    best = (0.0, None)
    m = len(r)
    for L in (1,2,3):
        for i in range(1, m-1-L):
            block = r[i:i+L]
            remain = r[:i] + r[i+L:]
            for j in range(1, len(remain)):
                if j == i or j == i+1: pass
                nr = remain[:j] + block + remain[j:]
                if not recompute_arrivals_safe(inst, nr):
                    continue
                delta = route_cost(inst.dist, np.array(nr, dtype=np.int32)) - base
                if delta < best[0]: best = (delta, nr)
    if best[1] is not None and best[0] < -1e-9:
        return True, best[1]
    return False, r

def two_opt_intra(inst: Instance, r: List[int]) -> Tuple[bool, List[int]]:
    if len(r) <= 4: return False, r
    base = route_cost(inst.dist, np.array(r, dtype=np.int32))
    best = (0.0, None)
    for i in range(1, len(r)-2):
        for j in range(i+1, len(r)-1):
            nr = r[:i] + r[i:j+1][::-1] + r[j+1:]
            if not recompute_arrivals_safe(inst, nr):
                continue
            delta = route_cost(inst.dist, np.array(nr, dtype=np.int32)) - base
            if delta < best[0]: best = (delta, nr)
    if best[1] is not None and best[0] < -1e-9:
        return True, best[1]
    return False, r

def relocate_inter(inst: Instance, r1: List[int], r2: List[int],
                   allowed_sets: Optional[List[Set[int]]] = None) -> Tuple[bool, List[int], List[int]]:
    if len(r1) <= 3: return False, r1, r2
    base = route_cost(inst.dist, np.array(r1, dtype=np.int32)) + route_cost(inst.dist, np.array(r2, dtype=np.int32))
    pos_u_list = list(range(1, len(r1)-1)); random.shuffle(pos_u_list)
    for pos_u in pos_u_list:
        u = r1[pos_u]
        remain1 = r1[:pos_u] + r1[pos_u+1:]
        if not capacity_ok(inst, remain1): continue
        pos_ins_list = list(range(1, len(r2))); random.shuffle(pos_ins_list)
        for pos_ins in pos_ins_list:
            if allowed_sets is not None:
                allowed = allowed_sets[u]
                left, right = r2[pos_ins-1], r2[pos_ins]
                if (left not in allowed) and (right not in allowed):
                    continue
            nr2 = r2[:pos_ins] + [u] + r2[pos_ins:]
            if not capacity_ok(inst, nr2): continue
            if not (recompute_arrivals_safe(inst, remain1) and recompute_arrivals_safe(inst, nr2)): continue
            c = route_cost(inst.dist, np.array(remain1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32))
            if c < base - 1e-9:
                return True, remain1, nr2
    return False, r1, r2

def swap_11(inst: Instance, r1: List[int], r2: List[int],
            allowed_sets: Optional[List[Set[int]]] = None) -> Tuple[bool, List[int], List[int]]:
    if len(r1) <= 3 or len(r2) <= 3: return False, r1, r2
    base = route_cost(inst.dist, np.array(r1, dtype=np.int32)) + route_cost(inst.dist, np.array(r2, dtype=np.int32))
    idxs1 = list(range(1, len(r1)-1)); random.shuffle(idxs1)
    for i in idxs1:
        u = r1[i]
        idxs2 = list(range(1, len(r2)-1)); random.shuffle(idxs2)
        for j in idxs2:
            v = r2[j]
            if allowed_sets is not None:
                au = allowed_sets[u]; av = allowed_sets[v]
                if (v not in au) and (u not in av):
                    continue
            nr1 = r1[:i] + [v] + r1[i+1:]
            nr2 = r2[:j] + [u] + r2[j+1:]
            if not (capacity_ok(inst, nr1) and capacity_ok(inst, nr2)): continue
            if not (recompute_arrivals_safe(inst, nr1) and recompute_arrivals_safe(inst, nr2)): continue
            c = route_cost(inst.dist, np.array(nr1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32))
            if c < base - 1e-9:
                return True, nr1, nr2
    return False, r1, r2

def two_opt_star(inst: Instance, r1: List[int], r2: List[int],
                 allowed_sets: Optional[List[Set[int]]] = None) -> Tuple[bool, List[int], List[int]]:
    if len(r1) <= 3 or len(r2) <= 3: return False, r1, r2
    base = route_cost(inst.dist, np.array(r1, dtype=np.int32)) + route_cost(inst.dist, np.array(r2, dtype=np.int32))
    cut1 = list(range(1, len(r1)-1)); random.shuffle(cut1)
    for i in cut1:
        cut2 = list(range(1, len(r2)-1)); random.shuffle(cut2)
        for j in cut2:
            if allowed_sets is not None:
                u = r1[i]; v = r2[j]
                au = allowed_sets[u]; av = allowed_sets[v]
                if (v not in au) and (u not in av):
                    continue
            nr1 = r1[:i] + r2[j:]
            nr2 = r2[:j] + r1[i:]
            if not (capacity_ok(inst, nr1) and capacity_ok(inst, nr2)): continue
            if not (recompute_arrivals_safe(inst, nr1) and recompute_arrivals_safe(inst, nr2)): continue
            c = route_cost(inst.dist, np.array(nr1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32))
            if c < base - 1e-9: return True, nr1, nr2
    return False, r1, r2

def rvnd(inst: Instance, routes: List[List[int]], max_loops: int = 5,
         nn: Optional[list[np.ndarray]] = None) -> List[List[int]]:
    if not routes: return routes

    # Préparer un set des voisins autorisés pour CHAQUE noeud (1 seule fois)
    allowed_sets: Optional[List[Set[int]]] = None
    if nn is not None:
        allowed_sets = [set(neigh.tolist()) for neigh in nn]

    loops = 0
    while loops < max_loops:
        improved_any = False
        # Intra
        for r_idx in range(len(routes)):
            changed, nr = or_opt_intra(inst, routes[r_idx])
            if changed: routes[r_idx] = nr; improved_any = True
        for r_idx in range(len(routes)):
            changed, nr = two_opt_intra(inst, routes[r_idx])
            if changed: routes[r_idx] = nr; improved_any = True
        # Inter
        for a in range(len(routes)):
            for b in range(len(routes)):
                if a == b: continue
                changed, ra, rb = relocate_inter(inst, routes[a], routes[b], allowed_sets)
                if changed: routes[a], routes[b] = ra, rb; improved_any = True
                changed, ra, rb = swap_11(inst, routes[a], routes[b], allowed_sets)
                if changed: routes[a], routes[b] = ra, rb; improved_any = True
                changed, ra, rb = two_opt_star(inst, routes[a], routes[b], allowed_sets)
                if changed: routes[a], routes[b] = ra, rb; improved_any = True
        if not improved_any: break
        loops += 1
    return [r for r in routes if len(r) > 2]

# ------------------------------ HGS core (parallélisé) ------------------------------
def tour_from_routes(routes: List[List[int]]) -> List[int]:
    out: List[int] = []
    for r in routes:
        out.extend(r[1:-1])
    return out

def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1)
    if n == 0: return []
    s1 = set(p1)
    p2f = [x for x in p2 if x in s1]
    missing = [x for x in p1 if x not in p2f]
    p2n = p2f + missing
    a = np.random.randint(0, n); b = np.random.randint(0, n)
    if a > b: a, b = b, a
    child = [-1]*n
    child[a:b+1] = p1[a:b+1]
    used = set(child[a:b+1])
    pos = (b+1) % n
    for x in p2n:
        if x in used: continue
        while child[pos] != -1:
            pos = (pos + 1) % n
        child[pos] = x
        used.add(x)
    return child

def shaw_relatedness(inst: Instance, a: int, b: int) -> float:
    return inst.dist[a,b]

def shaw_removal(inst: Instance, routes: List[List[int]], q: int) -> List[int]:
    allc = [c for r in routes for c in r[1:-1]]
    if not allc: return []
    seed = int(np.random.choice(np.array(allc)))
    removed = {seed}
    while len(removed) < min(q, len(allc)):
        cand = min((c for c in allc if c not in removed), key=lambda x: shaw_relatedness(inst, seed, x))
        removed.add(cand)
    return list(removed)

def random_removal(inst: Instance, routes: List[List[int]], q: int) -> List[int]:
    allc = [c for r in routes for c in r[1:-1]]
    np.random.shuffle(allc)
    return allc[:q]

def remove_customers(routes: List[List[int]], rem: List[int]) -> List[List[int]]:
    S = set(rem); new = []
    for r in routes:
        nr = [0] + [c for c in r[1:-1] if c not in S] + [0]
        if len(nr) > 2: new.append(nr)
    return new

def regret_repair(inst: Instance, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    if not routes: routes = [[0,0]]
    arrs: List[List[float]] = []
    for r in routes:
        ok, arr = recompute_arrivals(inst, r)
        arrs.append(arr if ok else [0.0]*len(r))
    unserved = set(removed)
    while unserved:
        best = None; best_reg = -1e18
        for u in list(unserved):
            choices = []
            for ridx, r in enumerate(routes):
                arr = arrs[ridx]
                b1 = (1e18, None, None); b2 = (1e18, None, None)
                for pos in range(1, len(r)):
                    feas, delta, new_arr = feasible_insert(inst, r, pos, u, arr)
                    if not feas: continue
                    if delta < b1[0]: b2 = b1; b1 = (delta, pos, new_arr)
                    elif delta < b2[0]: b2 = (delta, pos, new_arr)
                if b1[1] is not None:
                    regret = b2[0] - b1[0] if np.isfinite(b2[0]) else b1[0]
                    choices.append((regret, b1[0], ridx, b1[1], b1[2]))
            if choices:
                choices.sort(reverse=True)
                if choices[0][0] > best_reg:
                    best = (u, choices[0]); best_reg = choices[0][0]
        if best is None:
            u = unserved.pop()
            nr = [0, u, 0]
            ok, arr = recompute_arrivals(inst, nr)
            if not ok: arr = [0.0, max(inst.dist[0,u], inst.ready[u]), 0.0]
            routes.append(nr); arrs.append(arr)
        else:
            u, (_, _, ridx, pos, new_arr) = best
            routes[ridx] = routes[ridx][:pos] + [u] + routes[ridx][pos:]
            arrs[ridx] = new_arr
            unserved.remove(u)
    return routes

# ------------------------------ Heuristiques de recommandation ------------------------------
def _estimate_dist_bytes(n: int) -> int:
    # Approx mémoire de la matrice dist (float64)
    return (n + 1) * (n + 1) * 8  # bytes

def _os_is_windows() -> bool:
    try:
        return platform.system().lower().startswith("win")
    except Exception:
        return os.name == "nt"

def recommend_workers_and_fast(inst: Instance,
                               time_limit: Optional[float],
                               cpu_count: Optional[int] = None) -> dict:
    """
    Heuristiques pour choisir workers / fast / nnk :
    - en fonction de n, famille (X-n***, Solomon...), budget temps, CPU, overhead Windows/memmap
    """
    n = int(inst.n)
    fam, num, grp = _classify_instance(inst)
    cpu = max(1, int(cpu_count or (os.cpu_count() or 1)))
    TL = time_limit if time_limit is not None else 60.0

    # Estimations overhead/mémoire
    dist_mb = _estimate_dist_bytes(n) / (1024 * 1024)

    # --- workers ---
    workers = min(cpu, 8)
    if TL <= 40:
        workers = min(workers, 6)
    if _os_is_windows():
        workers = min(workers, 6 if TL <= 60 or n <= 150 else 8)
    if fam == 'x' and n >= 300:
        workers = min(workers, 6)
    if dist_mb >= 700:
        workers = min(workers, 6)
    if n <= 90:
        workers = min(workers, 4 if TL <= 40 else 6)
    workers = max(1, workers)

    # --- fast / nnk ---
    if fam == 'x' or n >= 120:
        fast = True
    else:
        fast = True  # utile même pour n moyen ; nnk ajusté ci-dessous

    if n <= 80:
        nnk = 22
    elif n <= 200:
        nnk = 28
    elif n <= 400:
        nnk = 32
    else:
        nnk = 35
    if TL <= 30:
        nnk = max(18, nnk - 6)

    notes = (f"fam={fam}, n={n}, dist≈{dist_mb:.1f}MB, TL={TL:.0f}s → "
             f"workers={workers}, fast={fast}, nnk={nnk}")
    return dict(workers=workers, fast=fast, nnk=nnk, notes=notes)

# ------------------------------ HGS driver ------------------------------
def _seed_task(inst: Instance, nn: Optional[list[np.ndarray]], init: str, seed_val: int) -> List[List[int]]:
    random.seed(seed_val); np.random.seed(seed_val)
    routes = build_seed(inst) if init == 'regret' else build_seed_sweep(inst)
    return rvnd(inst, routes, max_loops=3, nn=nn)

def _random_tour_task(inst: Instance, nn: Optional[list[np.ndarray]], has_tw: bool, seed_val: int) -> Tuple[List[List[int]], List[int]]:
    random.seed(seed_val); np.random.seed(seed_val)
    nodes = list(range(1, inst.n+1))
    random.shuffle(nodes)
    tour = nodes.copy()
    routes = (split_vrptw(inst, tour) if has_tw else split_cvrp(inst, tour)) or [[0,u,0] for u in tour]
    routes = rvnd(inst, routes, max_loops=2, nn=nn)
    return routes, tour_from_routes(routes)

def _offspring_task(inst: Instance,
                    nn: Optional[list[np.ndarray]],
                    has_tw: bool,
                    p1: List[int], p2: List[int],
                    best_routes: List[List[int]],
                    seed_val: int) -> List[List[int]]:
    random.seed(seed_val); np.random.seed(seed_val)
    ctour = ox_crossover(p1, p2)
    croutes = split_vrptw(inst, ctour) if has_tw else split_cvrp(inst, ctour)
    if croutes is None:
        q = max(2, inst.n//20)
        removed = shaw_removal(inst, best_routes, q) if np.random.rand() < 0.7 else random_removal(inst, best_routes, q)
        cand = remove_customers(best_routes, removed)
        cand = regret_repair(inst, cand, removed)
        ctour = tour_from_routes(cand)
        croutes = (split_vrptw(inst, ctour) if has_tw else split_cvrp(inst, ctour)) or cand
    croutes = rvnd(inst, croutes, max_loops=3, nn=nn)
    if np.random.rand() < 0.30:
        q = max(2, inst.n//25)
        removed = shaw_removal(inst, croutes, q) if np.random.rand()<0.7 else random_removal(inst, croutes, q)
        cand = remove_customers(croutes, removed)
        cand = regret_repair(inst, cand, removed)
        cand = rvnd(inst, cand, max_loops=2, nn=nn)
        if total_cost(inst, cand) < total_cost(inst, croutes):
            croutes = cand
    return croutes

def hgs_solve(inst: Instance, time_loops: int, pop_size: int,
              init: str = 'auto', nn: Optional[list[np.ndarray]] = None,
              workers: int = 1, time_limit: Optional[float] = None,
              _joblib_opts: Optional[dict] = None,
              record_convergence: bool = False) -> Tuple[List[List[int]], List[float]]:
    """
    HGS principal.
    Si record_convergence=True, renvoie aussi une liste 'convergence' contenant
    le meilleur coût à chaque itération.
    """
    if init == 'auto':
        init = 'regret' if inst.has_tw else 'sweep'

    def fit(rs): 
        return total_cost(inst, rs)

    jl = dict(backend="loky", prefer="processes")
    if _joblib_opts:
        jl.update(_joblib_opts)

    pop_tours: List[List[int]] = []
    pop_routes: List[List[List[int]]] = []
    
    # --- Liste de convergence ---
    convergence: List[float] = []

    # ==========================
    # 1) Construction population initiale
    # ==========================
    if workers > 1 and HAVE_JOBLIB:
        seeds_to_build = min(pop_size, max(8, workers*2))
        base_seed = random.randint(1, 10_000_000)
        seed_vals = [base_seed + i for i in range(seeds_to_build)]
        seed_routes = Parallel(n_jobs=workers, **jl)(
            delayed(_seed_task)(inst, nn, init, sv) for sv in seed_vals
        )
        for rs in seed_routes:
            pop_routes.append(rs)
            pop_tours.append(tour_from_routes(rs))

        remain = max(0, pop_size - len(pop_tours))
        if remain > 0:
            seed_vals2 = [base_seed + 10_000 + i for i in range(remain)]
            extras = Parallel(n_jobs=workers, **jl)(
                delayed(_random_tour_task)(inst, nn, inst.has_tw, sv) for sv in seed_vals2
            )
            for rs, tour in extras:
                pop_routes.append(rs)
                pop_tours.append(tour)
    else:
        # Construction séquentielle
        for _ in range(min(pop_size, 8)):
            routes = build_seed(inst) if init == 'regret' else build_seed_sweep(inst)
            routes = rvnd(inst, routes, max_loops=3, nn=nn)
            pop_routes.append(routes)
            pop_tours.append(tour_from_routes(routes))

        nodes = list(range(1, inst.n+1))
        for _ in range(max(0, pop_size - len(pop_tours))):
            random.shuffle(nodes)
            tour = nodes.copy()
            routes = (split_vrptw(inst, tour) if inst.has_tw else split_cvrp(inst, tour)) or [[0, u, 0] for u in tour]
            routes = rvnd(inst, routes, max_loops=2, nn=nn)
            pop_routes.append(routes)
            pop_tours.append(tour_from_routes(routes))

    # ==========================
    # 2) Best initial + enregistrer toutes les solutions initiales
    # ==========================
    best = min(pop_routes, key=fit)
    bestc = fit(best)
    
    # Enregistrer la progression pendant la construction initiale
    if record_convergence:
        all_costs = [fit(rs) for rs in pop_routes]
        all_costs.sort()  # Trier pour voir la progression
        convergence.extend(all_costs)

    batch_size = 1
    if workers > 1 and HAVE_JOBLIB:
        batch_size = max(workers * 2, 8)
    print(f"[parallel] workers={workers}, batch={batch_size}")

    t_start = time.perf_counter()

    # ==========================
    # 3) Boucle HGS
    # ==========================
    for _ in range(time_loops):
        if time_limit is not None and (time.perf_counter() - t_start) >= time_limit:
            break

        if batch_size == 1:
            # Mode séquentiel
            i, j = np.random.choice(len(pop_tours), 2, replace=False)
            ctour = ox_crossover(pop_tours[i], pop_tours[j])
            croutes = split_vrptw(inst, ctour) if inst.has_tw else split_cvrp(inst, ctour)
            if croutes is None:
                q = max(2, inst.n//20)
                removed = shaw_removal(inst, best, q) if np.random.rand() < 0.7 else random_removal(inst, best, q)
                cand = remove_customers(best, removed)
                cand = regret_repair(inst, cand, removed)
                ctour = tour_from_routes(cand)
                croutes = (split_vrptw(inst, ctour) if inst.has_tw else split_cvrp(inst, ctour)) or cand
            croutes = rvnd(inst, croutes, max_loops=3, nn=nn)
            cc = fit(croutes)
            if np.random.rand() < 0.30:
                q = max(2, inst.n//25)
                removed = shaw_removal(inst, croutes, q) if np.random.rand() < 0.7 else random_removal(inst, croutes, q)
                cand = remove_customers(croutes, removed)
                cand = regret_repair(inst, cand, removed)
                cand = rvnd(inst, cand, max_loops=2, nn=nn)
                c2 = fit(cand)
                if c2 < cc:
                    croutes, cc = cand, c2
            pop_routes.append(croutes)
            pop_tours.append(tour_from_routes(croutes))
            if len(pop_routes) > pop_size:
                worst_idx = int(np.argmax([fit(r) for r in pop_routes]))
                pop_routes.pop(worst_idx)
                pop_tours.pop(worst_idx)
            if cc < bestc - 1e-9:
                best = croutes
                bestc = cc
                # Enregistrer immédiatement les améliorations
                if record_convergence:
                    convergence.append(bestc)

        else:
            # Mode parallèle
            idx_pairs = []
            for __ in range(batch_size):
                i, j = np.random.choice(len(pop_tours), 2, replace=False)
                idx_pairs.append((i, j))
            base_seed = random.randint(1, 10_000_000)
            seeds = [base_seed + k for k in range(batch_size)]
            children = Parallel(n_jobs=workers, **jl)(
                delayed(_offspring_task)(
                    inst, nn, inst.has_tw,
                    pop_tours[i], pop_tours[j],
                    best, seeds[k]
                )
                for k, (i, j) in enumerate(idx_pairs)
            )
            for croutes in children:
                cc = fit(croutes)
                pop_routes.append(croutes)
                pop_tours.append(tour_from_routes(croutes))
                if len(pop_routes) > pop_size:
                    worst_idx = int(np.argmax([fit(r) for r in pop_routes]))
                    pop_routes.pop(worst_idx)
                    pop_tours.pop(worst_idx)
                if cc < bestc - 1e-9:
                    best = croutes
                    bestc = cc
                    # Enregistrer immédiatement les améliorations
                    if record_convergence:
                        convergence.append(bestc)

        # --- Enregistrer la convergence après chaque itération ---
        if record_convergence:
            # Enregistrer le meilleur coût actuel de la population
            current_best = min([fit(rs) for rs in pop_routes])
            convergence.append(current_best)
            # Mettre à jour bestc si nécessaire
            if current_best < bestc:
                bestc = current_best

    return best, convergence

# ------------------------------ .sol reference (gap) ------------------------------
def parse_sol_routes_and_cost(sol_path: Path) -> Tuple[Optional[float], List[List[int]]]:
    cost = None
    routes: List[List[int]] = []
    if not sol_path.exists():
        return None, []
    txt = sol_path.read_text(encoding='utf-8', errors='ignore')
    for ln in txt.splitlines():
        m = re.search(r'(?i)\bcost\b\s*[: ]\s*([0-9]+(?:\.[0-9]+)?)', ln)
        if m:
            try:
                cost = float(m.group(1))
            except Exception:
                pass
        r = re.search(r'(?i)\broute\b\s*#?\s*\d+\s*:\s*(.*)$', ln.strip())
        if r:
            seq = [int(x) for x in re.findall(r'\d+', r.group(1))]
            routes.append(seq)
    return cost, routes

def compute_routes_cost_with_instance(inst: Instance, plain_routes: List[List[int]]) -> float:
    total = 0.0
    for seq in plain_routes:
        route = [0] + seq + [0]
        total += route_cost(inst.dist, np.array(route, dtype=np.int32))
    return total

def find_sol_reference_cost(inst: Instance, inst_path: Path) -> Optional[float]:
    stem = inst_path.stem
    cand = inst_path.with_suffix('.sol')
    cost, routes = parse_sol_routes_and_cost(cand)
    if cost is not None:
        return cost
    if routes:
        return compute_routes_cost_with_instance(inst, routes)
    data_root = (Path.cwd() / 'data').resolve()
    if data_root.exists():
        for p in data_root.rglob(stem + '.sol'):
            cost, routes = parse_sol_routes_and_cost(p)
            if cost is not None:
                return cost
            if routes:
                return compute_routes_cost_with_instance(inst, routes)
    return None

# ------------------------------ Classification & Auto-params ------------------------------
def stable_seed_from_name(name: str) -> int:
    h = hashlib.md5(name.encode('utf-8')).hexdigest()
    return (int(h[:2], 16) % 3) + 1

def _classify_instance(inst: Instance) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Retourne (family, number, group) où :
    - family ∈ {'solomon','x','ortec','cvrplib','other'}
    - number : pour Solomon, le numéro (101..)
    - group : pour Solomon, 'C'/'R'/'RC' ; sinon None
    """
    stem = Path(inst.name).stem.lower()
    m = re.match(r'^(c|r|rc)\s*[-_]?(\d+)$', stem)
    if m:
        grp = m.group(1).upper()
        num = int(m.group(2))
        return ('solomon', num, grp)
    if stem.startswith('x-n'):
        return ('x', None, None)
    if 'ortec' in stem:
        return ('ortec', None, None)
    if stem.endswith('.vrp') or any(k in stem for k in ('-n', '_n')):
        return ('cvrplib', None, None)
    return ('other', None, None)

def auto_params(inst: Instance) -> dict:
    n = inst.n
    name = inst.name.lower()
    fam, num, grp = _classify_instance(inst)

    init = 'regret' if inst.has_tw else 'sweep'
    is_x = (fam == 'x')
    fast = (n >= 120) or is_x

    if n <= 80: nnk = 20
    elif n <= 200: nnk = 25
    elif n <= 400: nnk = 30
    else: nnk = 35

    if n <= 50:         loops, pop = 400, 48
    elif n <= 120:      loops, pop = 600, 56
    elif n <= 300:      loops, pop = 800, 64
    elif n <= 600:      loops, pop = 900, 72
    else:               loops, pop = 1200, 80

    time_limit = None
    cpu = max(1, os.cpu_count() or 1)
    workers = min(cpu, 8)

    if fam == 'solomon':
        is_set1 = (num is not None and num < 200)
        init = 'regret'
        if n <= 120:
            nnk = max(nnk, 25)
        if is_set1:
            time_limit = 30 if n <= 100 else 45
        else:
            time_limit = 60 if n <= 150 else 75
        if grp in ('R','RC'):
            workers = min(cpu, 8)
        else:
            workers = min(cpu, 6 if n <= 80 else 8)
    elif fam == 'x':
        fast = True
        nnk = max(nnk, 30)
        if n <= 200:      time_limit = 60
        elif n <= 400:    time_limit = 90
        elif n <= 700:    time_limit = 120
        else:             time_limit = 180
        workers = min(cpu, 6 if n >= 300 else 8)
    elif fam == 'ortec':
        fast = True
        nnk = max(nnk, 30)
        time_limit = 120 if n <= 300 else 180
        workers = min(cpu, 8)
    else:
        if not inst.has_tw:
            init = 'sweep'
        time_limit = 45 if n <= 120 else (75 if n <= 300 else 120)
        workers = min(cpu, 8 if n >= 80 else 4)

    if is_x and n >= 500:
        loops = max(loops, 900)
        pop   = max(pop, 72)

    # >>> Appliquer les recommandations fines
    rec = recommend_workers_and_fast(inst, time_limit=time_limit)
    workers = rec['workers']
    fast = rec['fast']
    nnk = rec['nnk']
    notes = rec.get('notes', '')

    seed = stable_seed_from_name(name)
    return dict(
        loops=loops, pop=pop, seed=seed, fast=fast, nnk=nnk, init=init,
        workers=workers, time_limit=time_limit, notes=notes
    )

# ------------------------------ Validation & IO ------------------------------
def validate_solution(inst: Instance, routes: List[List[int]]) -> None:
    seen: List[int] = []
    for r in routes: seen += r[1:-1]
    if len(seen) != inst.n or len(set(seen)) != inst.n:
        missing = sorted(list(set(range(1, inst.n+1)) - set(seen)))
        dups    = sorted({x for x in seen if seen.count(x) > 1})
        extra   = sorted(list(set(seen) - set(range(1, inst.n+1))))
        raise ValueError(f'Solution invalide: manquants={missing}, doublons={dups}, extras={extra}')
    for idx, r in enumerate(routes, 1):
        if not capacity_ok(inst, r):
            raise ValueError(f'Capacité dépassée sur la route #{idx}')
        if inst.has_tw:
            ok, _ = recompute_arrivals(inst, r)
            if not ok: raise ValueError(f'Fenêtre de temps violée sur la route #{idx}')

def solve_file(path: str,
               loops: Optional[int] = None,
               pop: Optional[int] = None,
               seed: Optional[int] = None,
               fast: Optional[bool] = None,
               nnk: Optional[int] = None,
               init: str = 'auto',
               workers: Optional[int] = None,
               time_limit: Optional[float] = None,
               return_raw: bool = False
               ) -> Union[str, Tuple[float, float, List[float]]]:

    path_l = path.lower()
    inst_path = Path(path)
    inst = parse_solomon_txt(path) if path_l.endswith('.txt') else parse_cvrplib_vrp(path)

    # Auto-paramètres
    ap = auto_params(inst)
    loops = ap['loops'] if loops is None else loops
    pop   = ap['pop']   if pop   is None else pop
    seed  = ap['seed']  if seed  is None else seed
    fast  = ap['fast']  if fast  is None else fast
    nnk   = ap['nnk']   if nnk   is None else nnk
    init  = ('regret' if inst.has_tw else 'sweep') if init == 'auto' else init
    workers    = ap['workers']    if workers    is None else workers
    time_limit = ap['time_limit'] if time_limit is None else time_limit
    notes = ap.get('notes', '')

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if workers is None:
        try:
            workers = max(1, os.cpu_count() or 1)
        except Exception:
            workers = 1

    nn = build_nn(inst.dist, nnk) if fast else None

    print(f"[auto] {inst.name}: I={loops}, P={pop}, S={seed}, fast={fast}, nnk={nnk}, init={init}, W={workers}, T={time_limit}")
    if notes:
        print(f"        {notes}")

    if workers > 1 and not HAVE_JOBLIB:
        print("[warn] joblib indisponible -> workers=1 (série)")
        workers = 1

    joblib_tmp = (Path.cwd() / ".joblib_memmap")
    joblib_tmp.mkdir(exist_ok=True)

    loader = ConsoleLoader()
    t0 = time.perf_counter()
    loader.start()
    try:
        routes, convergence = hgs_solve(
            inst,
            time_loops=loops,
            pop_size=pop,
            init=init,
            nn=nn,
            workers=workers,
            time_limit=time_limit,
            _joblib_opts=dict(
                backend="loky",
                prefer="processes",
                temp_folder=str(joblib_tmp),
                max_nbytes="10M",
                batch_size="auto",
            ),
            record_convergence=True,
        )
    finally:
        loader.stop()
    elapsed = time.perf_counter() - t0

    # Validation
    validate_solution(inst, routes)

    # Temps du plus long trajet
    max_route_time = 0.0
    for r in routes:
        ok, arr = recompute_arrivals(inst, r)
        if ok and arr:
            rt = arr[-1]
        else:
            rt = route_cost(inst.dist, np.array(r, dtype=np.int32))
        if rt > max_route_time:
            max_route_time = rt

    # Texte de sortie classique
    lines = []
    for i, r in enumerate(routes, 1):
        lines.append(f"Route #{i}: " + " ".join(str(x) for x in r[1:-1]))
    cost = total_cost(inst, routes)
    lines.append(f"Cost {int(round(cost))}")
    lines.append(f"Time {elapsed:.2f}s")
    lines.append(f"Temps du plus long trajet {max_route_time:.2f}s")

    # GAP (si .sol trouvé)
    ref = find_sol_reference_cost(inst, inst_path)
    if ref and ref > 0:
        gap = 100.0 * (cost - ref) / ref
        lines.append(f"gap {gap:.2f}% (ref {int(round(ref))})")
    else:
        gap = float('nan')
        lines.append("gap N/A")

    if return_raw:
        # Retour purement numérique pour scripts externes
        return cost, gap, convergence
    else:
        # Comportement original : texte pour affichage
        return "\n".join(lines)
# ------------------------------ Analyse répertoire ------------------------------
def find_candidate_files(root: Path, max_depth: int = 4) -> Tuple[List[Path], List[Path]]:
    root = root.resolve()
    tw: List[Path] = []
    vrp: List[Path] = []
    def rec(dirp: Path, d: int):
        if d > max_depth: return
        try:
            for e in dirp.iterdir():
                if e.is_dir():
                    rec(e, d+1)
                else:
                    suf = e.suffix.lower()
                    if suf == '.txt':
                        tw.append(e.resolve())
                    elif suf == '.vrp':
                        vrp.append(e.resolve())
        except PermissionError:
            pass
    rec(root, 0)
    tw = sorted(set(tw), key=lambda p: str(p).lower())
    vrp = sorted(set(vrp), key=lambda p: str(p).lower())
    return tw, vrp

def analyze_dir(root: str, match: Optional[str] = None) -> None:
    """
    Parcourt un dossier, détecte .vrp/.txt, parse chaque instance,
    et affiche la recommandation workers/fast/nnk/time_limit avec un résumé.
    """
    rootp = Path(root).resolve()
    if not rootp.exists():
        print(f"[analyze] Dossier introuvable: {root}")
        return
    tw, vrp = find_candidate_files(rootp)
    files = tw + vrp
    if not files:
        print(f"[analyze] Aucun .vrp/.txt sous {rootp}")
        return

    print("Nom                          n   fam       W  fast  nnk   T(s)  distMB   Notes")
    print("-" * 90)
    for p in sorted(files, key=lambda x: x.name.lower()):
        if match and (match.lower() not in p.name.lower()):
            continue
        try:
            inst = parse_solomon_txt(str(p)) if p.suffix.lower()=='.txt' else parse_cvrplib_vrp(str(p))
            ap = auto_params(inst)
            fam, _, _ = _classify_instance(inst)
            dist_mb = _estimate_dist_bytes(inst.n) / (1024 * 1024)
            print(f"{p.name:28} {inst.n:4d} {fam:8}  {ap['workers']:2d}  {str(ap['fast']):>4}  {ap['nnk']:3d}  "
                  f"{(ap['time_limit'] if ap['time_limit'] else 0):5.0f}  {dist_mb:7.1f}  {ap.get('notes','')}")
        except Exception as e:
            print(f"{p.name:28}  ERROR: {e}")

# ------------------------------ Menu interactif (split TW / non-TW) ------------------------------
def interactive_choose_split() -> List[Path]:
    project_root = Path.cwd()
    data_root = (project_root / 'data')
    root = data_root if data_root.exists() else project_root

    tw, vrp = find_candidate_files(root)
    if not tw and not vrp:
        print(f"Aucun .vrp/.txt trouvé sous {root}. Donne un chemin :")
        s = input("Fichier : ").strip()
        p = Path(s)
        return [p] if p.exists() else []

    while True:
        print("\n=== Sélection d'instances ===")
        print(f"[1] Avec TW (.txt, Solomon)   - {len(tw)} fichiers")
        print(f"[2] Sans TW (.vrp, CVRPLIB)   - {len(vrp)} fichiers")
        print("[Q] Quitter")
        choice = input("Choix catégorie [1/2/Q]: ").strip().lower()
        if choice in ('q', 'quit', 'exit'):
            return []
        if choice not in ('1','2'):
            continue

        pool = tw if choice == '1' else vrp
        if not pool:
            print("Aucun fichier dans cette catégorie.")
            continue
        print("\nFichiers détectés:")
        for i, p in enumerate(pool, 1):
            try:
                rel = p.relative_to(project_root)
            except Exception:
                rel = p
            print(f"  [{i}] {rel}")
        s = input("\nChoisis (ex: 1,3) ou chemin, ou 'b' pour revenir, ou 'q' : ").strip()
        if s.lower() in {'b', 'back'}:
            continue
        if s.lower() in {'q','quit','exit'}:
            return []
        if re.fullmatch(r"\d+(,\d+)*", s):
            idxs = [int(x) for x in s.split(',')]
            chosen: List[Path] = []
            for k in idxs:
                if 1 <= k <= len(pool): chosen.append(pool[k-1])
            return chosen
        p = Path(s)
        if p.exists():
            return [p]
        print("Entrée invalide, recommence.")

# ------------------------------ CLI ------------------------------
def main():
    parser = argparse.ArgumentParser(description='VRP/VRPTW HGS solver (menu + gap via .sol + auto-tuning + parallel + analyze)')
    parser.add_argument('files', nargs='*', help='paths to .vrp/.txt (sinon menu interactif)')
    parser.add_argument('-I','--iterations', type=int, default=None, help='HGS time loops (override auto)')
    parser.add_argument('-P','--popsize', type=int, default=None, help='Population size (override auto)')
    parser.add_argument('-S','--seed', type=int, default=None, help='Random seed (override auto)')
    parser.add_argument('--nnk', type=int, default=None, help='k plus-proches voisins (override auto)')
    parser.add_argument('--init', choices=['auto','regret','sweep'], default='auto', help='Heuristique init (override)')
    parser.add_argument('--fast', dest='fast', action='store_const', const=True, default=None, help='Force fast ON')
    parser.add_argument('--no-fast', dest='fast', action='store_const', const=False, help='Force fast OFF')
    parser.add_argument('-W','--workers', type=int, default=None, help="Nb de processus en parallèle (auto si non fourni)")
    parser.add_argument('-T','--time-limit', type=float, default=None, help='Limite temps en secondes pour la boucle HGS (auto si non fourni)')
    parser.add_argument('--analyze', type=str, default=None, help="Analyser un dossier (.vrp/.txt) et recommander W/fast/nnk")
    parser.add_argument('--match', type=str, default=None, help="Filtre sur le nom de fichier (sous-chaîne)")
    args = parser.parse_args()

    # Mode analyse : lister les fichiers et proposer les réglages
    if args.analyze:
        analyze_dir(args.analyze, match=args.match)
        sys.exit(0)

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        paths = interactive_choose_split()

    if not paths:
        sys.exit(0)

    for p in paths:
        print(f"=== {p} ===")
        try:
            print(solve_file(str(p),
                             loops=args.iterations,
                             pop=args.popsize,
                             seed=args.seed,
                             fast=args.fast,
                             nnk=args.nnk,
                             init=args.init,
                             workers=args.workers,
                             time_limit=args.time_limit))
        except Exception as e:
            import traceback
            print("Error:", e)
            traceback.print_exc()
        print()

def maybe_wait_for_exit() -> None:
    # When run from a double-clicked batch/script, keep the console open after printing results.
    if sys.stdin and sys.stdin.isatty():
        try:
            input("\nAppuyez sur Entrée pour fermer...")
        except EOFError:
            pass

if __name__ == '__main__':
    main()
    maybe_wait_for_exit()
