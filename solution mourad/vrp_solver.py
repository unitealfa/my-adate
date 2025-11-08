# vrp_solver.py
# Minimal VRP/VRPTW solver with regret insertion + intra-route local search.
# Supports Solomon VRPTW .txt and CVRPLIB .vrp (no TW).
# NEW: Interactive file picker if no CLI args (choose .txt/.vrp anywhere).

import re, math, sys, os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

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

def parse_solomon_txt(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    num = cap = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("VEHICLE"):
            for j in range(i+1, min(i+6, len(lines))):
                m = re.findall(r"(\d+)\s+(\d+)", lines[j])
                if m:
                    num, cap = map(int, m[0]); break
            break
    cust_start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("CUSTOMER"):
            cust_start = i + 1
            break
    data = []
    for ln in lines[cust_start:]:
        if re.match(r"^\d+", ln):
            parts = re.split(r"\s+", ln.strip())
            if len(parts) >= 7:
                cid, x, y, dem, ready, due, service = parts[:7]
                data.append((int(cid), float(x), float(y), float(dem), float(ready), float(due), float(service)))
    data = sorted(data, key=lambda t: t[0])
    n = len(data) - 1
    coords = np.zeros((n+1, 2), dtype=float)
    demand = np.zeros(n+1, dtype=float)
    ready = np.zeros(n+1, dtype=float)
    due = np.zeros(n+1, dtype=float)
    service = np.zeros(n+1, dtype=float)
    for cid, x, y, dem, r, d, s in data:
        coords[cid] = (x, y); demand[cid] = dem; ready[cid] = r; due[cid] = d; service[cid] = s
    dist = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(-1))
    return Instance(Path(path).name, n, coords, demand, ready, due, service,
                    int(cap) if cap else 10**9, int(num) if num else n, dist, True)

def parse_cvrplib_vrp(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    name = None; capacity = None; dimension = None
    node_coord_start = demand_start = depot_start = None
    for i, ln in enumerate(lines):
        U = ln.upper()
        if U.startswith('NAME'): name = ln.split(':')[-1].strip()
        elif U.startswith('DIMENSION'): dimension = int(ln.split(':')[-1].strip())
        elif U.startswith('CAPACITY'): capacity = int(ln.split(':')[-1].strip())
        elif U.startswith('NODE_COORD_SECTION'): node_coord_start = i+1
        elif U.startswith('DEMAND_SECTION'): demand_start = i+1
        elif U.startswith('DEPOT_SECTION'): depot_start = i+1
    n_nodes = dimension
    coords = np.zeros((n_nodes+1, 2), dtype=float)
    i = node_coord_start
    while i < len(lines) and not lines[i].upper().endswith('SECTION') and not lines[i].upper().startswith('DEPOT_SECTION'):
        parts = re.split(r"\s+", lines[i])
        if len(parts) >= 3 and parts[0].isdigit():
            idx = int(parts[0]); coords[idx] = (float(parts[1]), float(parts[2]))
        i += 1
    demand = np.zeros(n_nodes+1, dtype=float)
    i = demand_start
    while i < len(lines):
        if lines[i].upper().startswith('DEPOT_SECTION'): break
        parts = re.split(r"\s+", lines[i])
        if len(parts) >= 2 and parts[0].isdigit():
            idx = int(parts[0]); demand[idx] = float(parts[1])
        i += 1
    depot_id = 1
    i = depot_start
    while i < len(lines):
        if lines[i].startswith('-1'): break
        if lines[i].isdigit(): depot_id = int(lines[i]); break
        i += 1
    cust_ids = [i for i in range(1, n_nodes+1) if i != depot_id]
    n = len(cust_ids)
    coords_remap = np.zeros((n+1, 2), dtype=float)
    demand_remap = np.zeros(n+1, dtype=float)
    coords_remap[0] = coords[depot_id]
    for new_idx, orig in enumerate(cust_ids, start=1):
        coords_remap[new_idx] = coords[orig]; demand_remap[new_idx] = demand[orig]
    dist = np.sqrt(((coords_remap[:,None,:] - coords_remap[None,:,:])**2).sum(-1))
    ready = np.zeros(n+1, dtype=float); due = np.full(n+1, 1e9, dtype=float); service = np.zeros(n+1, dtype=float)
    fname = Path(path).name; m = re.search(r'-k(\d+)', fname)
    if m: k = int(m.group(1))
    else: k = math.ceil(demand_remap[1:].sum() / capacity) if capacity else n
    return Instance(fname, n, coords_remap, demand_remap, ready, due, service,
                    int(capacity) if capacity else 10**9, int(k), dist, False)

def travel_time(inst: Instance, i: int, j: int) -> float:
    return inst.dist[i, j]

def recompute_arrivals(inst: Instance, route: List[int]):
    arr = [0.0]
    for idx in range(1, len(route)):
        prev = route[idx-1]; cur = route[idx]
        depart_prev = max(arr[idx-1], inst.ready[prev]) + inst.service[prev]
        arr_cur = depart_prev + travel_time(inst, prev, cur)
        if arr_cur < inst.ready[cur]: arr_cur = inst.ready[cur]
        if arr_cur > inst.due[cur] + 1e-9: return False, []
        arr.append(arr_cur)
    return True, arr

def feasible_insert_route(inst: Instance, route: List[int], pos: int, u: int, load: float, arr_times: List[float]):
    if load + inst.demand[u] > inst.capacity: return (False, math.inf, [])
    i = route[pos-1]; j = route[pos]
    delta = travel_time(inst, i, u) + travel_time(inst, u, j) - travel_time(inst, i, j)
    new_arr = arr_times[:pos]
    new_route = route[:pos] + [u] + route[pos:]
    for idx in range(pos, len(new_route)):
        prev = new_route[idx-1]; cur = new_route[idx]
        depart_prev = max(new_arr[idx-1], inst.ready[prev]) + inst.service[prev]
        arr_cur = depart_prev + travel_time(inst, prev, cur)
        if arr_cur < inst.ready[cur]: arr_cur = inst.ready[cur]
        if arr_cur > inst.due[cur] + 1e-9: return (False, math.inf, [])
        new_arr.append(arr_cur)
    return (True, delta, new_arr)

def build_initial_solution(inst: Instance):
    unserved = set(range(1, inst.n+1))
    routes = []; loads = []; arrivals = []
    while unserved:
        best_inserts = []
        for r_idx, route in enumerate(routes):
            load = loads[r_idx]; arr = arrivals[r_idx]
            for u in unserved:
                best1 = (math.inf, None, None); best2 = (math.inf, None, None)
                for pos in range(1, len(route)):
                    feas, delta, new_arr = feasible_insert_route(inst, route, pos, u, load, arr)
                    if not feas: continue
                    if delta < best1[0]:
                        best2 = best1; best1 = (delta, pos, new_arr)
                    elif delta < best2[0]:
                        best2 = (delta, pos, new_arr)
                if best1[1] is not None:
                    regret = best2[0] - best1[0] if math.isfinite(best2[0]) else best1[0]
                    best_inserts.append((u, r_idx, best1[1], best1[0], best1[2], regret))
        if best_inserts:
            best_inserts.sort(key=lambda t: (-t[5], t[3]))
            u, r_idx, pos, delta, new_arr, _ = best_inserts[0]
            routes[r_idx] = routes[r_idx][:pos] + [u] + routes[r_idx][pos:]
            arrivals[r_idx] = new_arr
            loads[r_idx] += inst.demand[u]
            unserved.remove(u)
        else:
            if len(routes) >= inst.k:
                # force new single-customer route
                chosen_u = None; best_delta = math.inf
                for u in unserved:
                    feas, _, _ = feasible_insert_route(inst, [0,0], 1, u, 0.0, [0.0, 0.0])
                    if feas:
                        delta = travel_time(inst, 0, u) + travel_time(inst, u, 0)
                        if delta < best_delta: best_delta = delta; chosen_u = u
                if chosen_u is None: chosen_u = next(iter(unserved))
                routes.append([0, chosen_u, 0]); loads.append(inst.demand[chosen_u])
                arrivals.append([0.0, max(0.0 + travel_time(inst, 0, chosen_u), inst.ready[chosen_u]), 0.0])
                unserved.remove(chosen_u)
            else:
                if inst.has_tw:
                    seed = min(unserved, key=lambda u: inst.ready[u])
                else:
                    seed = max(unserved, key=lambda u: (inst.demand[u], travel_time(inst, 0, u)))
                routes.append([0, seed, 0]); loads.append(inst.demand[seed])
                depart0 = 0.0
                arr_u = max(depart0 + travel_time(inst, 0, seed), inst.ready[seed])
                arrivals.append([0.0, arr_u, 0.0])
                unserved.remove(seed)
    return routes, loads

def route_cost(inst: Instance, route: List[int]) -> float:
    return sum(inst.dist[route[i], route[i+1]] for i in range(len(route)-1))

def total_cost(inst: Instance, routes: List[List[int]]) -> float:
    return sum(route_cost(inst, r) for r in routes)

def local_search_light(inst: Instance, routes: List[List[int]], max_iters: int = 120) -> List[List[int]]:
    improved = True; iters = 0
    while improved and iters < max_iters:
        improved = False; iters += 1
        # intra relocate
        for r_idx, r in enumerate(routes):
            if len(r) <= 3: continue
            base = route_cost(inst, r)
            best = (0.0, None)
            for i in range(1, len(r)-1):
                u = r[i]
                for j in range(1, len(r)):
                    if j == i or j == i+1: continue
                    new_r = r[:i] + r[i+1:]
                    new_r = new_r[:j] + [u] + new_r[j:]
                    ok, _ = recompute_arrivals(inst, new_r)
                    if not ok: continue
                    delta = route_cost(inst, new_r) - base
                    if delta < best[0]: best = (delta, new_r)
            if best[1] is not None and best[0] < -1e-9:
                routes[r_idx] = best[1]; improved = True
        if improved: continue
        # 2-opt intra
        for r_idx, r in enumerate(routes):
            if len(r) <= 4: continue
            base = route_cost(inst, r); best = (0.0, None)
            for i in range(1, len(r)-2):
                for j in range(i+1, len(r)-1):
                    new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
                    ok, _ = recompute_arrivals(inst, new_r)
                    if not ok: continue
                    delta = route_cost(inst, new_r) - base
                    if delta < best[0]: best = (delta, new_r)
            if best[1] is not None and best[0] < -1e-9:
                routes[r_idx] = best[1]; improved = True
    return [r for r in routes if len(r) > 2]

BEST_KNOWN = {'A-n32-k5.vrp': 784.0, 'X-n101-k25.vrp': 27591.0}

def solve(path: str) -> str:
    p = str(path)
    if p.lower().endswith('.txt'): inst = parse_solomon_txt(p)
    else: inst = parse_cvrplib_vrp(p)
    routes, _ = build_initial_solution(inst)
    routes = local_search_light(inst, routes, max_iters=120)
    lines = []
    for idx, r in enumerate(routes, start=1):
        seq = " ".join(str(x) for x in r[1:-1])
        lines.append(f"Route #{idx}: {seq}")
    cost = total_cost(inst, routes); lines.append(f"Cost {int(round(cost))}")
    best = BEST_KNOWN.get(inst.name, None)
    if best: lines.append(f"gap {100.0*(cost-best)/best:.2f}%")
    else: lines.append("gap N/A")
    return "\n".join(lines)

# --------- Interactive picker ---------
def find_candidate_files(root: Path, exts=(".vrp", ".txt"), max_depth=3) -> List[Path]:
    root = root.resolve()
    out = []
    def rec(dirpath: Path, depth: int):
        if depth > max_depth: return
        try:
            for e in dirpath.iterdir():
                if e.is_dir():
                    rec(e, depth+1)
                else:
                    if e.suffix.lower() in exts:
                        out.append(e)
        except PermissionError:
            pass
    rec(root, 0)
    # unique + sort
    out = sorted(set(out), key=lambda p: str(p).lower())
    return out

def interactive_choose() -> List[Path]:
    cwd = Path.cwd()
    candidates = find_candidate_files(cwd)
    print("\nAucun argument fourni. Sélectionne les fichiers à résoudre.")
    if candidates:
        print("\nFichiers détectés :")
        for i, p in enumerate(candidates, 1):
            rel = p.relative_to(cwd)
            print(f"  [{i}] {rel}")
        print("\nEntre :")
        print("  - un ou plusieurs numéros séparés par des virgules (ex: 1,3,5)")
        print("  - OU un chemin vers un fichier (ex: data/C101.txt)")
        print("  - OU 'q' pour quitter")
        s = input("\nTon choix : ").strip()
        if s.lower() in {"q", "quit", "exit"}:
            return []
        # numbers?
        if re.fullmatch(r"\d+(,\d+)*", s):
            idxs = [int(x) for x in s.split(",")]
            chosen = []
            for k in idxs:
                if 1 <= k <= len(candidates):
                    chosen.append(candidates[k-1])
                else:
                    print(f"Indice hors limite: {k}")
            return chosen
        else:
            p = Path(s).expanduser()
            if p.exists() and p.suffix.lower() in {".vrp", ".txt"}:
                return [p.resolve()]
            else:
                print("Chemin invalide ou extension non reconnue.")
                return []
    else:
        print("Aucun .vrp/.txt trouvé. Entre un chemin de fichier (ou 'q') :")
        s = input("Fichier : ").strip()
        if s.lower() in {"q", "quit", "exit"}:
            return []
        p = Path(s).expanduser()
        if p.exists() and p.suffix.lower() in {".vrp", ".txt"}:
            return [p.resolve()]
        print("Chemin invalide.")
        return []

# --------------- CLI ---------------
if __name__ == "__main__":
    # If args given -> solve them. Else -> interactive picker.
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        paths = interactive_choose()
    if not paths:
        sys.exit(0)
    for p in paths:
        try:
            print(f"\n=== {p} ===")
            print(solve(str(p)))
        except Exception as e:
            print(f"Error on {p}: {e}")
    print()
