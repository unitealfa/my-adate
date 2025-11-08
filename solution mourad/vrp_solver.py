
import math, re, sys
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
        lines = [ln.strip() for ln in f if ln.strip()]
    num = cap = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("VEHICLE"):
            for j in range(i+1, min(i+7, len(lines))):
                m = re.findall(r"(\d+)\s+(\d+)", lines[j])
                if m:
                    num, cap = map(int, m[0]); break
            break
    cust_start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("CUSTOMER"):
            cust_start = i+1; break
    data = []
    for ln in lines[cust_start:]:
        if re.match(r"^\d+", ln):
            parts = re.split(r"\s+", ln)
            if len(parts) >= 7:
                cid, x, y, dem, ready, due, service = parts[:7]
                data.append((int(cid), float(x), float(y), float(dem), float(ready), float(due), float(service)))
    data.sort(key=lambda t: t[0])
    n = len(data)-1
    coords = np.zeros((n+1,2), dtype=np.float64)
    demand = np.zeros(n+1); ready = np.zeros(n+1); due = np.zeros(n+1); service = np.zeros(n+1)
    for cid, x, y, dem, r, d, s in data:
        coords[cid] = (x,y); demand[cid] = dem; ready[cid] = r; due[cid] = d; service[cid] = s
    dist = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
    return Instance(Path(path).name, n, coords, demand, ready, due, service, int(cap) if cap else 10**9, int(num) if num else n, dist, True)

def parse_cvrplib_vrp(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    name = None; cap = None; dim = None
    node_start = dem_start = depot_start = None
    for i, ln in enumerate(lines):
        U = ln.upper()
        if U.startswith('NAME'): name = ln.split(':')[-1].strip()
        elif U.startswith('DIMENSION'): dim = int(ln.split(':')[-1].strip())
        elif U.startswith('CAPACITY'): cap = int(ln.split(':')[-1].strip())
        elif U.startswith('NODE_COORD_SECTION'): node_start = i+1
        elif U.startswith('DEMAND_SECTION'): dem_start = i+1
        elif U.startswith('DEPOT_SECTION'): depot_start = i+1
    coords = np.zeros((dim+1,2), dtype=np.float64)
    i = node_start
    while i < len(lines) and not lines[i].upper().endswith('SECTION') and not lines[i].upper().startswith('DEPOT_SECTION'):
        parts = re.split(r"\s+", lines[i])
        if len(parts) >= 3 and parts[0].isdigit():
            idx = int(parts[0]); coords[idx] = (float(parts[1]), float(parts[2]))
        i += 1
    demand = np.zeros(dim+1)
    i = dem_start
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
    cust_ids = [i for i in range(1, dim+1) if i != depot_id]
    n = len(cust_ids)
    coords2 = np.zeros((n+1,2), dtype=np.float64); demand2 = np.zeros(n+1)
    coords2[0] = coords[depot_id]
    for new, orig in enumerate(cust_ids, 1):
        coords2[new] = coords[orig]; demand2[new] = demand[orig]
    dist = np.sqrt(((coords2[:,None,:]-coords2[None,:,:])**2).sum(-1))
    ready = np.zeros(n+1); due = np.full(n+1, 1e9); service = np.zeros(n+1)
    fname = Path(path).name; m = re.search(r'-k(\d+)', fname)
    if m: k = int(m.group(1))
    else: k = math.ceil(demand2[1:].sum()/cap) if cap else n
    return Instance(fname, n, coords2, demand2, ready, due, service, int(cap) if cap else 10**9, int(k), dist, False)

@njit(cache=True)
def route_cost(dist: np.ndarray, route: np.ndarray) -> float:
    c = 0.0
    for i in range(route.shape[0]-1):
        c += dist[route[i], route[i+1]]
    return c

def total_cost(inst: Instance, routes: List[List[int]]) -> float:
    return sum(route_cost(inst.dist, np.array(r, dtype=np.int32)) for r in routes)

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

def split_tw(inst: Instance, tour: List[int]) -> Optional[List[List[int]]]:
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
            if arr_u > inst.due[u]+1e-9: ok = False
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
    while cur>0:
        i = prev[cur]
        routes.append([0]+tour[i:cur]+[0])
        cur = i
    routes.reverse()
    return routes

def feasible_insert(inst: Instance, route: List[int], pos: int, u: int, load: float, arr_times: List[float]):
    if load + inst.demand[u] > inst.capacity: return False, 1e18, []
    i = route[pos-1]; j = route[pos]
    delta = inst.dist[i,u]+inst.dist[u,j]-inst.dist[i,j]
    new_arr = arr_times[:pos]
    new_route = route[:pos]+[u]+route[pos:]
    for idx in range(pos, len(new_route)):
        prev = new_route[idx-1]; cur = new_route[idx]
        depart_prev = max(new_arr[idx-1], inst.ready[prev]) + inst.service[prev]
        arr_cur = depart_prev + inst.dist[prev, cur]
        if arr_cur < inst.ready[cur]: arr_cur = inst.ready[cur]
        if arr_cur > inst.due[cur]+1e-9: return False, 1e18, []
        new_arr.append(arr_cur)
    return True, delta, new_arr

def build_seed(inst: Instance) -> List[List[int]]:
    unserved = set(range(1, inst.n+1))
    routes = []; loads = []; arrivals = []
    while unserved:
        best = []
        for r_idx, r in enumerate(routes):
            load = loads[r_idx]; arr = arrivals[r_idx]
            for u in list(unserved):
                b1 = (1e18, None, None); b2 = (1e18, None, None)
                for pos in range(1, len(r)):
                    feas, delta, new_arr = feasible_insert(inst, r, pos, u, load, arr)
                    if not feas: continue
                    if delta < b1[0]:
                        b2 = b1; b1 = (delta, pos, new_arr)
                    elif delta < b2[0]:
                        b2 = (delta, pos, new_arr)
                if b1[1] is not None:
                    regret = b2[0]-b1[0] if np.isfinite(b2[0]) else b1[0]
                    best.append((u, r_idx, b1[1], b1[0], b1[2], regret))
        if best:
            best.sort(key=lambda t: (-t[5], t[3]))
            u,r_idx,pos,_,new_arr,_ = best[0]
            routes[r_idx] = routes[r_idx][:pos]+[u]+routes[r_idx][pos:]
            arrivals[r_idx] = new_arr
            loads[r_idx] += inst.demand[u]
            unserved.remove(u)
        else:
            if len(routes) >= inst.k:
                # open single
                u = next(iter(unserved))
                routes.append([0,u,0]); loads.append(inst.demand[u])
                arrivals.append([0.0, max(inst.dist[0,u], inst.ready[u]), 0.0])
                unserved.remove(u)
            else:
                seed = min(unserved, key=lambda u: inst.ready[u]) if inst.has_tw else max(unserved, key=lambda u: (inst.demand[u], inst.dist[0,u]))
                routes.append([0,seed,0]); loads.append(inst.demand[seed])
                arrivals.append([0.0, max(inst.dist[0,seed], inst.ready[seed]), 0.0])
                unserved.remove(seed)
    return routes

def recompute_arrivals_safe(inst: Instance, r: List[int]) -> bool:
    ok,_ = recompute_arrivals(inst, r); return ok

def rvnd(inst: Instance, routes: List[List[int]], max_loops: int = 5) -> List[List[int]]:
    loops = 0
    while loops < max_loops:
        improved = False
        # relocate inter
        best = (0.0, None)
        for rf in range(len(routes)):
            r1 = routes[rf]
            if len(r1)<=3: continue
            for pos_u in range(1, len(r1)-1):
                u = r1[pos_u]
                for rt in range(len(routes)):
                    r2 = routes[rt]
                    for pos_ins in range(1, len(r2)):
                        if rf==rt and (pos_ins==pos_u or pos_ins==pos_u+1): continue
                        nr1 = r1[:pos_u]+r1[pos_u+1:]
                        nr2 = r2[:pos_ins]+[u]+r2[pos_ins:]
                        if not (recompute_arrivals_safe(inst, nr1) and recompute_arrivals_safe(inst, nr2)):
                            continue
                        delta = route_cost(inst.dist, np.array(nr1, dtype=np.int32)) + route_cost(inst.dist, np.array(nr2, dtype=np.int32)) - route_cost(inst.dist, np.array(r1, dtype=np.int32)) - route_cost(inst.dist, np.array(r2, dtype=np.int32))
                        if delta < best[0]: best = (delta, (rf,rt,pos_u,pos_ins,nr1,nr2))
        if best[1] is not None and best[0] < -1e-9:
            rf,rt,pos_u,pos_ins,nr1,nr2 = best[1]
            routes[rf]=nr1; routes[rt]=nr2; improved=True

        # relocate intra
        if not improved:
            for r_idx,r in enumerate(routes):
                if len(r)<=3: continue
                base = route_cost(inst.dist, np.array(r, dtype=np.int32))
                best=(0.0,None)
                for i in range(1,len(r)-1):
                    u=r[i]
                    for j in range(1,len(r)):
                        if j==i or j==i+1: continue
                        nr=r[:i]+r[i+1:]
                        nr=nr[:j]+[u]+nr[j:]
                        if not recompute_arrivals_safe(inst, nr): continue
                        delta = route_cost(inst.dist, np.array(nr, dtype=np.int32)) - base
                        if delta < best[0]: best=(delta,nr)
                if best[1] is not None and best[0] < -1e-9:
                    routes[r_idx]=best[1]; improved=True

        # 2-opt intra
        if not improved:
            for r_idx,r in enumerate(routes):
                if len(r)<=4: continue
                base = route_cost(inst.dist, np.array(r, dtype=np.int32))
                best=(0.0,None)
                for i in range(1,len(r)-2):
                    for j in range(i+1,len(r)-1):
                        nr=r[:i]+r[i:j+1][::-1]+r[j+1:]
                        if not recompute_arrivals_safe(inst, nr): continue
                        delta = route_cost(inst.dist, np.array(nr, dtype=np.int32)) - base
                        if delta < best[0]: best=(delta,nr)
                if best[1] is not None and best[0] < -1e-9:
                    routes[r_idx]=best[1]; improved=True
        if not improved: break
        loops += 1
    return [r for r in routes if len(r)>2]

def tour_from_routes(routes: List[List[int]]) -> List[int]:
    out=[]; [out.extend(r[1:-1]) for r in routes]; return out

def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n=len(p1)
    if n==0: return []
    a=np.random.randint(0,n); b=np.random.randint(0,n)
    if a>b: a,b=b,a
    child=[-1]*n; child[a:b+1]=p1[a:b+1]
    pos=(b+1)%n; idx=(b+1)%n; used=set(child[a:b+1])
    while -1 in child:
        if p2[idx] not in used:
            child[pos]=p2[idx]; pos=(pos+1)%n; used.add(p2[idx])
        idx=(idx+1)%n
    return child

def shaw_relatedness(inst: Instance, a: int, b: int) -> float:
    return inst.dist[a,b]

def shaw_removal(inst: Instance, routes: List[List[int]], q: int) -> List[int]:
    allc = [c for r in routes for c in r[1:-1]]
    if not allc: return []
    seed = int(np.random.choice(allc))
    removed={seed}
    while len(removed)<min(q,len(allc)):
        cand = min((c for c in allc if c not in removed), key=lambda x: shaw_relatedness(inst, seed, x))
        removed.add(cand)
    return list(removed)

def random_removal(inst: Instance, routes: List[List[int]], q: int) -> List[int]:
    allc=[c for r in routes for c in r[1:-1]]
    np.random.shuffle(allc); return allc[:q]

def remove_customers(routes: List[List[int]], rem: List[int]) -> List[List[int]]:
    S=set(rem); new=[]
    for r in routes:
        nr=[0]+[c for c in r[1:-1] if c not in S]+[0]
        if len(nr)>2: new.append(nr)
    return new

def feasible_insert(inst: Instance, route: List[int], pos: int, u: int, load: float, arr_times: List[float]):
    if load + inst.demand[u] > inst.capacity: return False, 1e18, []
    i = route[pos-1]; j = route[pos]
    delta = inst.dist[i,u]+inst.dist[u,j]-inst.dist[i,j]
    new_arr = arr_times[:pos]
    new_route = route[:pos]+[u]+route[pos:]
    for idx in range(pos, len(new_route)):
        prev = new_route[idx-1]; cur = new_route[idx]
        depart_prev = max(new_arr[idx-1], inst.ready[prev]) + inst.service[prev]
        arr_cur = depart_prev + inst.dist[prev, cur]
        if arr_cur < inst.ready[cur]: arr_cur = inst.ready[cur]
        if arr_cur > inst.due[cur]+1e-9: return False, 1e18, []
        new_arr.append(arr_cur)
    return True, delta, new_arr

def regret_repair(inst: Instance, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    if not routes: routes=[[0,0]]
    loads=[sum(inst.demand[c] for c in r) for r in routes]
    arrs=[]
    for r in routes:
        ok,arr=recompute_arrivals(inst,r); arrs.append(arr if ok else [0.0]*len(r))
    unserved=set(removed)
    while unserved:
        best=None; best_reg=-1e18
        for u in list(unserved):
            choices=[]
            for ridx,r in enumerate(routes):
                load=loads[ridx]; arr=arrs[ridx]
                b1=(1e18,None,None); b2=(1e18,None,None)
                for pos in range(1,len(r)):
                    feas,delta,new_arr = feasible_insert(inst,r,pos,u,load,arr)
                    if not feas: continue
                    if delta<b1[0]:
                        b2=b1; b1=(delta,pos,new_arr)
                    elif delta<b2[0]:
                        b2=(delta,pos,new_arr)
                if b1[1] is not None:
                    regret = b2[0]-b1[0] if np.isfinite(b2[0]) else b1[0]
                    choices.append((regret,b1[0],ridx,b1[1],b1[2]))
            if choices:
                choices.sort(reverse=True)
                if choices[0][0] > best_reg:
                    best=(u,choices[0]); best_reg=choices[0][0]
        if best is None:
            u=unserved.pop()
            routes.append([0,u,0]); loads.append(inst.demand[u]); arrs.append([0.0,max(inst.dist[0,u],inst.ready[u]),0.0])
        else:
            u,(regret,delta,ridx,pos,new_arr)=best
            routes[ridx]=routes[ridx][:pos]+[u]+routes[ridx][pos:]
            arrs[ridx]=new_arr; loads[ridx]+=inst.demand[u]; unserved.remove(u)
    return routes

def hgs_solve(inst: Instance, time_loops: int = 200, pop_size: int = 30) -> List[List[int]]:
    pop_tours=[]; pop_routes=[]
    for _ in range(min(pop_size,8)):
        routes=build_seed(inst)
        routes=rvnd(inst,routes,max_loops=3)
        pop_routes.append(routes); pop_tours.append(tour_from_routes(routes))
    nodes=list(range(1,inst.n+1))
    for _ in range(max(0,pop_size-len(pop_tours))):
        np.random.shuffle(nodes); tour=nodes.copy()
        routes = split_tw(inst,tour) or [[0,u,0] for u in tour]
        routes = rvnd(inst,routes,max_loops=2)
        pop_routes.append(routes); pop_tours.append(tour_from_routes(routes))
    def fit(rs): return total_cost(inst, rs)
    best=min(pop_routes,key=fit); bestc=fit(best)
    for _ in range(time_loops):
        i,j=np.random.choice(len(pop_tours),2,replace=False)
        ctour=ox_crossover(pop_tours[i],pop_tours[j])
        croutes=split_tw(inst,ctour)
        if croutes is None:
            removed=shaw_removal(inst,best,max(2,inst.n//20))
            cand=remove_customers(best,removed)
            cand=regret_repair(inst,cand,removed)
            ctour=tour_from_routes(cand)
            croutes=split_tw(inst,ctour) or cand
        croutes=rvnd(inst,croutes,max_loops=3)
        cc=fit(croutes)
        if np.random.rand()<0.25:
            q=max(2,inst.n//25)
            removed = shaw_removal(inst,croutes,q) if np.random.rand()<0.7 else random_removal(inst,croutes,q)
            cand=remove_customers(croutes,removed)
            cand=regret_repair(inst,cand,removed)
            cand=rvnd(inst,cand,max_loops=2)
            c2=fit(cand)
            if c2<cc: croutes=cand; cc=c2
        pop_routes.append(croutes); pop_tours.append(tour_from_routes(croutes))
        if len(pop_routes)>pop_size:
            worst_idx = int(np.argmax([fit(r) for r in pop_routes]))
            pop_routes.pop(worst_idx); pop_tours.pop(worst_idx)
        if cc<bestc-1e-9: best=croutes; bestc=cc
    return best

BEST_KNOWN={'A-n32-k5.vrp':784.0,'X-n101-k25.vrp':27591.0}

def solve_file(path: str) -> str:
    if path.lower().endswith('.txt'): inst=parse_solomon_txt(path)
    else: inst=parse_cvrplib_vrp(path)
    routes=hgs_solve(inst,time_loops=120,pop_size=28)
    lines=[]
    for i,r in enumerate(routes,1):
        lines.append(f"Route #{i}: "+" ".join(str(x) for x in r[1:-1]))
    cost=total_cost(inst,routes); lines.append(f"Cost {int(round(cost))}")
    best=BEST_KNOWN.get(inst.name,None)
    if best and best>0: lines.append(f"gap {100.0*(cost-best)/best:.2f}%")
    else: lines.append("gap N/A")
    return "\\n".join(lines)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python vrp_hgs_solver.py <file1> [file2 ...]")
        sys.exit(0)
    for p in sys.argv[1:]:
        print(f"=== {p} ===")
        try:
            print(solve_file(p))
        except Exception as e:
            print("Error:", e)
        print()
