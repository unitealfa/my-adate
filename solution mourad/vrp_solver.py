#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRP / VRPTW solver (HGS + Split(TW) + RVND + ALNS, Numba-accelerated when available)
- Reads CVRPLIB .vrp (no TW) and Solomon .txt (with TW)
- Supports k identical vehicles, capacity, and time windows (waiting allowed, no service outside window)
- Uses EUC_2D rounded distances for CVRPLIB (matches official costs)
- Interactive picker: choose files from ./data if no CLI args

Suggested params for competitive results (small/medium):
    python vrp_hgs_pro.py data/A-n32-k5.vrp -I 400 -P 48 -S 1
Large instances (~1000 customers):
    python vrp_hgs_pro.py data/X-n101-k25.vrp -I 800 -P 64 -S 1

Outputs:
    Route #i: <sequence>
    Cost <total>
    gap <xx.xx%>   (if best known is provided)

Author: you
"""

import math, re, sys, argparse, random, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

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

# ------------------------------ Nearest Neighbors (granular) ------------------------------
def build_nn(dist: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Pour chaque noeud u, calcule les k plus proches voisins (hors lui-même et hors dépôt 0).
    Retourne une liste de tableaux d'indices.
    """
    n = dist.shape[0]
    order = np.argsort(dist, axis=1)
    nn: list[np.ndarray] = []
    for u in range(n):
        cand = [int(v) for v in order[u] if v != u and v != 0][:k]
        nn.append(np.array(cand, dtype=np.int32))
    return nn

# ------------------------------ Best-known loader (optional) ------------------------------
def load_best_known(path: Optional[str]) -> dict:
    """
    Charge un mapping {nom_instance: best_cost} depuis:
      - JSON (dict)  ou
      - CSV / texte lignes:  name,cost   ou   name cost
    Tout est optionnel: si le fichier n'existe pas / invalide -> {}.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    txt = p.read_text(encoding='utf-8', errors='ignore').strip()
    # Essai JSON
    try:
        import json
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    # Fallback CSV / lignes "name cost"
    m = {}
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        parts = re.split(r'[\,\s;]+', ln)
        if len(parts) >= 2:
            name = parts[0]
            try:
                val = float(parts[1])
            except Exception:
                continue
            m[name] = val
    return m

# ------------------------------ Parsing ------------------------------
def parse_solomon_txt(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]
    # Clean and keep non-empty
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

    name = get_val('NAME') or Path(path).name
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

    coords_map = {}
    i = node_idx + 1
    while i < len(lines) and not U[i].endswith('SECTION') and not U[i].startswith('DEPOT_SECTION') and not U[i].startswith('DEMAND_SECTION'):
        parts = re.split(r"\s+", lines[i])
        if len(parts) >= 3 and parts[0].isdigit():
            idx = int(parts[0]); x = float(parts[1]); y = float(parts[2]); coords_map[idx] = (x,y)
        i += 1

    demand_map = {}
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

    # EUC_2D rounded distances (official CVRPLIB convention)
    diff = coords2[:, None, :] - coords2[None, :, :]
    dist = np.rint(np.sqrt((diff**2).sum(-1))).astype(np.float64)

    fname = Path(path).name
    m = re.search(r'-K(\d+)', fname, flags=re.IGNORECASE)
    if m: k = int(m.group(1))
    else: k = math.ceil(demand2[1:].sum()/capacity) if capacity>0 else n

    cap = int(capacity) if capacity>0 else 10**9
    return Instance(fname, n, coords2, demand2, ready, due, service, cap, int(k), dist, False)

# ------------------------------ Utilities ------------------------------
@njit(cache=True)
def route_cost(dist: np.ndarray, route: np.ndarray) -> float:
    c = 0.0
    for i in range(route.shape[0]-1):
        c += dist[route[i], route[i+1]]
    return c

def total_cost(inst: Instance, routes: List[List[int]]) -> float:
    return sum(route_cost(inst.dist, np.array(r, dtype=np.int32)) for r in routes)

# recompute arrival times respecting TW and waiting; returns (feasible?, arrivals)
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

# quick feasibility for capacity-only (CVRP)
def capacity_ok(inst: Instance, route: List[int]) -> bool:
    load = 0.0
    for c in route[1:-1]:
        load += inst.demand[c]
        if load > inst.capacity + 1e-9:
            return False
    return True

# ------------------------------ Split (DP) ------------------------------
# Split for CVRP (no TW)
def split_cvrp(inst: Instance, tour: List[int]) -> Optional[List[List[int]]]:
    n = len(tour)
    # Precompute cumulative demand and cumulative distances for quick evaluation
    cum_dem = np.zeros(n+1)
    for i in range(n):
        cum_dem[i+1] = cum_dem[i] + inst.demand[tour[i]]

    # cost of [i..j] as a route (0 -> tour[i] ... tour[j] -> 0), capacity respected
    def route_cost_ij(i, j) -> float:
        load = cum_dem[j+1] - cum_dem[i]
        if load > inst.capacity + 1e-9:
            return 1e18
        # 0 -> tour[i]
        c = inst.dist[0, tour[i]]
        # inside
        for t in range(i, j):
            c += inst.dist[tour[t], tour[t+1]]
        # -> 0
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
    if prev[n] == -1:
        return None
    routes = []
    cur = n
    while cur > 0:
        i = prev[cur]
        routes.append([0]+tour[i:cur]+[0])
        cur = i
    routes.reverse()
    return routes

# Split with TW (VRPTW)
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
            # return to depot
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
    # Capacity quick check (single insertion)
    load = sum(inst.demand[c] for c in route[1:-1]) + inst.demand[u]
    if load > inst.capacity + 1e-9: return False, 1e18, []

    i = route[pos-1]; j = route[pos]
    delta = inst.dist[i,u] + inst.dist[u,j] - inst.dist[i,j]

    # TW propagate
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
            # open new route with an appropriate seed (early TW or far/demand for CVRP)
            if inst.has_tw:
                seed = min(unserved, key=lambda u: inst.ready[u])
            else:
                seed = max(unserved, key=lambda u: (inst.demand[u], inst.dist[0,u]))
            r = [0, seed, 0]
            ok, arr = recompute_arrivals(inst, r)
            if not ok:
                # fallback: single customer route will always be feasible here
                arr = [0.0, max(inst.dist[0,seed], inst.ready[seed]), 0.0]
            routes.append(r); arrivals.append(arr)
            unserved.remove(seed)
    return routes

def build_seed_sweep(inst: Instance) -> List[List[int]]:
    """Construction rapide pour CVRP: tri par angle autour du dépôt + remplissage capacitaire."""
    depot = inst.coords[0]
    vec = inst.coords[1:] - depot
    ang = np.arctan2(vec[:,1], vec[:,0])
    order = np.argsort(ang) + 1
    routes: List[List[int]] = []
    cur: List[int] = [0]
    load = 0.0
    for u in order:
        d = float(inst.demand[int(u)])
        if load + d <= inst.capacity + 1e-9:
            cur.append(int(u)); load += d
        else:
            cur.append(0)
            routes.append(cur)
            cur = [0, int(u)]; load = d
    cur.append(0); routes.append(cur)
    return routes

# ------------------------------ RVND ------------------------------
def recompute_arrivals_safe(inst: Instance, r: List[int]) -> bool:
    ok, _ = recompute_arrivals(inst, r)
    return ok

# Intra-route or-opt (move chains of length L = 1,2,3 inside same route)
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

# Intra 2-opt

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

# Inter relocate (move single node between routes)

def relocate_inter(inst: Instance, r1: List[int], r2: List[int], nn: Optional[list[np.ndarray]] = None) -> Tuple[bool, List[int], List[int]]:
    if len(r1) <= 3: return False, r1, r2
    base = route_cost(inst.dist, np.array(r1, dtype=np.int32)) + route_cost(inst.dist, np.array(r2, dtype=np.int32))
    pos_u_list = list(range(1, len(r1)-1)); random.shuffle(pos_u_list)
    for pos_u in pos_u_list:
        u = r1[pos_u]
        remain1 = r1[:pos_u] + r1[pos_u+1:]
        if not capacity_ok(inst, remain1):
            continue
        pos_ins_list = list(range(1, len(r2))); random.shuffle(pos_ins_list)
        for pos_ins in pos_ins_list:
            if nn is not None:
                allowed = set(nn[u].tolist())
                left, right = r2[pos_ins-1], r2[pos_ins]
                if (left not in allowed) and (right not in allowed):
                    continue
            nr2 = r2[:pos_ins] + [u] + r2[pos_ins:]
            if not capacity_ok(inst, nr2):
                continue
            if not (recompute_arrivals_safe(inst, remain1) and recompute_arrivals_safe(inst, nr2)):
                continue
            c = route_cost(inst.dist, np.array(remain1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32))
            if c < base - 1e-9:
                return True, remain1, nr2
    return False, r1, r2

# Inter swap(1,1)

def swap_11(inst: Instance, r1: List[int], r2: List[int], nn: Optional[list[np.ndarray]] = None) -> Tuple[bool, List[int], List[int]]:
    if len(r1) <= 3 or len(r2) <= 3: return False, r1, r2
    base = route_cost(inst.dist, np.array(r1, dtype=np.int32)) + route_cost(inst.dist, np.array(r2, dtype=np.int32))
    idxs1 = list(range(1, len(r1)-1)); random.shuffle(idxs1)
    for i in idxs1:
        u = r1[i]
        idxs2 = list(range(1, len(r2)-1)); random.shuffle(idxs2)
        for j in idxs2:
            v = r2[j]
            if nn is not None:
                close_u = set(nn[u].tolist()); close_v = set(nn[v].tolist())
                if (v not in close_u) and (u not in close_v):
                    continue
            nr1 = r1[:i] + [v] + r1[i+1:]
            nr2 = r2[:j] + [u] + r2[j+1:]
            if not (capacity_ok(inst, nr1) and capacity_ok(inst, nr2)):
                continue
            if not (recompute_arrivals_safe(inst, nr1) and recompute_arrivals_safe(inst, nr2)):
                continue
            c = route_cost(inst.dist, np.array(nr1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32))
            if c < base - 1e-9:
                return True, nr1, nr2
    return False, r1, r2

# Inter 2-opt* (swap suffixes between two routes)

def two_opt_star(inst: Instance, r1: List[int], r2: List[int], nn: Optional[list[np.ndarray]] = None) -> Tuple[bool, List[int], List[int]]:
    if len(r1) <= 3 or len(r2) <= 3: return False, r1, r2
    base = route_cost(inst.dist, np.array(r1, dtype=np.int32)) + route_cost(inst.dist, np.array(r2, dtype=np.int32))
    cut1 = list(range(1, len(r1)-1)); random.shuffle(cut1)
    for i in cut1:
        u = r1[i]
        cut2 = list(range(1, len(r2)-1)); random.shuffle(cut2)
        for j in cut2:
            v = r2[j]
            if nn is not None:
                close_u = set(nn[u].tolist()); close_v = set(nn[v].tolist())
                if (v not in close_u) and (u not in close_v):
                    continue
            nr1 = r1[:i] + r2[j:]
            nr2 = r2[:j] + r1[i:]
            if not (capacity_ok(inst, nr1) and capacity_ok(inst, nr2)):
                continue
            if not (recompute_arrivals_safe(inst, nr1) and recompute_arrivals_safe(inst, nr2)):
                continue
            c = route_cost(inst.dist, np.array(nr1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32))
            if c < base - 1e-9:
                return True, nr1, nr2
    return False, r1, r2

# Full RVND over a set of neighborhoods

def rvnd(inst: Instance, routes: List[List[int]], max_loops: int = 5,
         nn: Optional[list[np.ndarray]] = None,
         time_limit: Optional[float] = None,
         t0: Optional[float] = None) -> List[List[int]]:
    if not routes:
        return routes
    loops = 0
    while loops < max_loops:
        improved = False
        neighborhoods = ['or', '2opt', 'relocate', 'swap', '2opt*']
        random.shuffle(neighborhoods)
        for nb in neighborhoods:
            if time_limit and t0 and (time.perf_counter() - t0) > time_limit:
                return [r for r in routes if len(r) > 2]
            if nb == 'or':
                for r_idx in range(len(routes)):
                    changed, nr = or_opt_intra(inst, routes[r_idx])
                    if changed:
                        routes[r_idx] = nr; improved = True; break
            elif nb == '2opt':
                for r_idx in range(len(routes)):
                    changed, nr = two_opt_intra(inst, routes[r_idx])
                    if changed:
                        routes[r_idx] = nr; improved = True; break
            elif nb == 'relocate':
                for a in range(len(routes)):
                    for b in range(len(routes)):
                        if a == b: continue
                        changed, ra, rb = relocate_inter(inst, routes[a], routes[b], nn)
                        if changed:
                            routes[a], routes[b] = ra, rb; improved = True; break
                    if improved: break
            elif nb == 'swap':
                for a in range(len(routes)):
                    for b in range(len(routes)):
                        if a == b: continue
                        changed, ra, rb = swap_11(inst, routes[a], routes[b], nn)
                        if changed:
                            routes[a], routes[b] = ra, rb; improved = True; break
                    if improved: break
            else:  # '2opt*'
                for a in range(len(routes)):
                    for b in range(len(routes)):
                        if a == b: continue
                        changed, ra, rb = two_opt_star(inst, routes[a], routes[b], nn)
                        if changed:
                            routes[a], routes[b] = ra, rb; improved = True; break
                    if improved: break
            if improved:
                break
        if not improved:
            break
        loops += 1
    return [r for r in routes if len(r) > 2]

# ------------------------------ HGS core ------------------------------
def tour_from_routes(routes: List[List[int]]) -> List[int]:
    out: List[int] = []
    for r in routes:
        out.extend(r[1:-1])
    return out

# Order Crossover with safety normalization

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

# ALNS: destroy (Shaw or random) + regret repair

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

# regret-k repair (k=2)

def regret_repair(inst: Instance, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    if not routes: routes = [[0,0]]
    # prepare arrival arrays
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
            # open a new route (single)
            u = unserved.pop()
            nr = [0, u, 0]
            ok, arr = recompute_arrivals(inst, nr)
            if not ok:
                arr = [0.0, max(inst.dist[0,u], inst.ready[u]), 0.0]
            routes.append(nr); arrs.append(arr)
        else:
            u, (regret, delta, ridx, pos, new_arr) = best
            routes[ridx] = routes[ridx][:pos] + [u] + routes[ridx][pos:]
            arrs[ridx] = new_arr
            unserved.remove(u)
    return routes

# Main HGS loop

def hgs_solve(inst: Instance, time_loops: int = 200, pop_size: int = 30,
              time_limit: Optional[float] = None,
              nn: Optional[list[np.ndarray]] = None,
              init: str = 'regret') -> List[List[int]]:
    t0 = time.perf_counter()
    # Initial population: mix greedy seeds and random tours
    pop_tours: List[List[int]] = []
    pop_routes: List[List[List[int]]] = []

    for _ in range(min(pop_size, 8)):
        if inst.has_tw or init == 'regret':
            routes = build_seed(inst)
        else:
            routes = build_seed_sweep(inst)
        routes = rvnd(inst, routes, max_loops=3, nn=nn, time_limit=time_limit, t0=t0)
        pop_routes.append(routes); pop_tours.append(tour_from_routes(routes))

    nodes = list(range(1, inst.n+1))
    for _ in range(max(0, pop_size - len(pop_tours))):
        random.shuffle(nodes)
        tour = nodes.copy()
        routes = (split_vrptw(inst, tour) if inst.has_tw else split_cvrp(inst, tour)) or [[0,u,0] for u in tour]
        routes = rvnd(inst, routes, max_loops=2, nn=nn, time_limit=time_limit, t0=t0)
        pop_routes.append(routes); pop_tours.append(tour_from_routes(routes))

    def fit(rs): return total_cost(inst, rs)
    best = min(pop_routes, key=fit); bestc = fit(best)

    for _ in range(time_loops):
        if time_limit and (time.perf_counter() - t0) > time_limit:
            break
        i, j = np.random.choice(len(pop_tours), 2, replace=False)
        ctour = ox_crossover(pop_tours[i], pop_tours[j])
        croutes = split_vrptw(inst, ctour) if inst.has_tw else split_cvrp(inst, ctour)
        if croutes is None:
            # ALNS recovery from current best
            q = max(2, inst.n//20)
            removed = shaw_removal(inst, best, q) if np.random.rand()<0.7 else random_removal(inst, best, q)
            cand = remove_customers(best, removed)
            cand = regret_repair(inst, cand, removed)
            ctour = tour_from_routes(cand)
            croutes = (split_vrptw(inst, ctour) if inst.has_tw else split_cvrp(inst, ctour)) or cand
        croutes = rvnd(inst, croutes, max_loops=3, nn=nn, time_limit=time_limit, t0=t0)
        cc = fit(croutes)

        # ALNS diversify occasionally
        if np.random.rand() < 0.30:
            q = max(2, inst.n//25)
            removed = shaw_removal(inst, croutes, q) if np.random.rand()<0.7 else random_removal(inst, croutes, q)
            cand = remove_customers(croutes, removed)
            cand = regret_repair(inst, cand, removed)
            cand = rvnd(inst, cand, max_loops=2, nn=nn, time_limit=time_limit, t0=t0)
            c2 = fit(cand)
            if c2 < cc: croutes, cc = cand, c2

        pop_routes.append(croutes); pop_tours.append(tour_from_routes(croutes))
        if len(pop_routes) > pop_size:
            worst_idx = int(np.argmax([fit(r) for r in pop_routes]))
            pop_routes.pop(worst_idx); pop_tours.pop(worst_idx)
        if cc < bestc - 1e-9:
            best = croutes; bestc = cc
    return best

# ------------------------------ Validation & IO ------------------------------
 


def validate_solution(inst: Instance, routes: List[List[int]]) -> None:
    seen: List[int] = []
    for r in routes: seen += r[1:-1]
    if len(seen) != inst.n or len(set(seen)) != inst.n:
        missing = sorted(list(set(range(1, inst.n+1)) - set(seen)))
        dups    = sorted({x for x in seen if seen.count(x) > 1})
        extra   = sorted(list(set(seen) - set(range(1, inst.n+1))))
        raise ValueError(f'Solution invalide: manquants={missing}, doublons={dups}, extras={extra}')
    # capacity and TW
    for idx, r in enumerate(routes, 1):
        if not capacity_ok(inst, r):
            raise ValueError(f'Capacité dépassée sur la route #{idx}')
        if inst.has_tw:
            ok, _ = recompute_arrivals(inst, r)
            if not ok: raise ValueError(f'Fenêtre de temps violée sur la route #{idx}')


def solve_file(path: str, loops: int, pop: int, seed: Optional[int]) -> str:
    # Exécute l’algo et imprime UNIQUEMENT le coût calculé
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    inst = parse_solomon_txt(path) if path.lower().endswith('.txt') else parse_cvrplib_vrp(path)
    routes = hgs_solve(inst, time_loops=loops, pop_size=pop)
    validate_solution(inst, routes)

    lines = []
    for i, r in enumerate(routes, 1):
        lines.append(f"Route #{i}: " + " ".join(str(x) for x in r[1:-1]))

    cost = total_cost(inst, routes)
    lines.append(f"Cost {int(round(cost))}")

    # Pas de gap, pas de best-known
    return "\n".join(lines)

# ------------------------------ Interactive Picker ------------------------------
def find_candidate_files(root: Path, exts=(".vrp", ".txt"), max_depth=3) -> List[Path]:
    root = root.resolve()
    out: List[Path] = []
    def rec(dirp: Path, d: int):
        if d > max_depth: return
        try:
            for e in dirp.iterdir():
                if e.is_dir(): rec(e, d+1)
                elif e.suffix.lower() in exts: out.append(e)
        except PermissionError:
            pass
    rec(root, 0)
    out = sorted(set(out), key=lambda p: str(p).lower())
    return out


def interactive_choose() -> List[Path]:
    project_root = Path.cwd()
    data_root = project_root / 'data'
    root = data_root if data_root.exists() else project_root
    cands = find_candidate_files(root)
    if not cands:
        print(f"Aucun .vrp/.txt dans {root}. Donne un chemin :")
        s = input("Fichier : ").strip()
        p = Path(s)
        return [p] if p.exists() else []
    print(f"\nFichiers détectés sous {root}:")
    for i, p in enumerate(cands, 1):
        try:
            rel = p.relative_to(project_root)
        except Exception:
            rel = p
        print(f"  [{i}] {rel}")
    s = input("\nChoisis (ex: 1,3) ou chemin, ou 'q' : ").strip()
    if s.lower() in {"q","quit","exit"}: return []
    if re.fullmatch(r"\d+(,\d+)*", s):
        idxs = [int(x) for x in s.split(',')]
        chosen = []
        for k in idxs:
            if 1 <= k <= len(cands): chosen.append(cands[k-1])
        return chosen
    p = Path(s)
    return [p] if p.exists() else []

# ------------------------------ CLI ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VRP/VRPTW HGS solver')
    parser.add_argument('files', nargs='*', help='paths to .vrp/.txt')
    parser.add_argument('-I','--iterations', type=int, default=200, help='HGS time loops')
    parser.add_argument('-P','--popsize', type=int, default=30, help='Population size')
    parser.add_argument('-S','--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--best-known', type=str, default=None, help="Chemin vers un fichier JSON/CSV contenant {name: best_cost}")
    parser.add_argument('-T','--time-limit', type=float, default=None, help='Temps max en secondes (arrêt propre)')
    parser.add_argument('--fast', action='store_true', help='Active granular neighborhoods + first-improvement')
    parser.add_argument('--nnk', type=int, default=25, help='Taille de la liste de plus-proches voisins (fast)')
    parser.add_argument('--init', choices=['regret','sweep'], default='regret', help="Heuristique d'initialisation (CVRP: 'sweep' conseillé; VRPTW: 'regret')")
    args = parser.parse_args()

    paths: List[Path]
    if not args.files:
        paths = interactive_choose()
    else:
        paths = [Path(p) for p in args.files]
    if not paths:
        sys.exit(0)

    for p in paths:
        print(f"=== {p} ===")
        try:
            print(solve_file(str(p),
                             loops=args.iterations,
                             pop=args.popsize,
                             seed=args.seed))
        except Exception as e:
            import traceback
            print("Error:", e)
            traceback.print_exc()
        print()
