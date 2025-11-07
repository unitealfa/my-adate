#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRPTW (fenÃƒÂªtres de temps) avec k vÃƒÂ©hicules identiques
Modes supportÃƒÂ©s :
  Ã¢â‚¬Â¢ SANS attente (noÃ¢â‚¬â€˜wait)
  Ã¢â‚¬Â¢ AVEC attente (wait)
  Ã¢â‚¬Â¢ HYBRIDE "JITÃ¢â‚¬â€˜WAIT" : sans attente par dÃƒÂ©faut, mais **autorise l'attente
    uniquement si cela rend une route faisable** (fallback local) Ã¢â‚¬â€ utile quand
    un client isolÃƒÂ© rend l'instance infaisable en noÃ¢â‚¬â€˜wait pur.

Objectif lexicographique :
  Phase A  Ã¢â€ â€™ minimiser le makespan T (retour du dernier camion)
  Phase B  Ã¢â€ â€™ minimiser la distance D sous la contrainte T Ã¢â€°Â¤ T*

ConÃƒÂ§u pour n Ã¢Ë†Ë† [10, 2000] avec auto-paramÃƒÂ©trage et journaux de progression.
Affiche ÃƒÂ  la fin :
  - RÃƒÂ©capitulatif (n, k, seed, T*, distance, gap)
  - Figure statique des tournÃƒÂ©es
  - Animation (matplotlib) des camions qui se dÃƒÂ©placent
  - Si JITÃ¢â‚¬â€˜WAIT a ÃƒÂ©tÃƒÂ© nÃƒÂ©cessaire : **liste des clients pour lesquels une attente a ÃƒÂ©tÃƒÂ© appliquÃƒÂ©e**

Correctifs & robustesse
-----------------------
Ã¢â‚¬Â¢ Correction rÃƒÂ©cursion infinie : `eval_route` ne s'appelle plus luiÃ¢â‚¬â€˜mÃƒÂªme.
Ã¢â‚¬Â¢ Ajout d'un **mode hybride JITÃ¢â‚¬â€˜WAIT** : si une ÃƒÂ©valuation noÃ¢â‚¬â€˜wait ÃƒÂ©choue,
  on retente la **mÃƒÂªme route** en mode wait et on **log** les clients qui ont
  effectivement nÃƒÂ©cessitÃƒÂ© une attente.
Ã¢â‚¬Â¢ Correction cornerÃ¢â‚¬â€˜cases dans les opÃƒÂ©rateurs de destroy.
Ã¢â‚¬Â¢ **NOUVEAU : Loader / barre de progression** pour ALNS et Tabu :
  - utilise `tqdm` si disponible ;
  - sinon, fallback en console avec pourcentage + ETA (une seule ligne mise ÃƒÂ  jour).

DÃƒÂ©pendances : numpy, matplotlib
Optionnel : tqdm (pour la barre de progression), vrplib (pour tests VRPLIB).
"""
from __future__ import annotations
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# tqdm (optionnel)
try:  # barre de progression si installÃƒÂ©e
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ===========================
#  Config globale
# ===========================

# Modes globaux basculÃƒÂ©s par run_solve
ALLOW_WAIT: bool = False           # tout en wait
ALLOW_JIT_WAIT: bool = True        # fallback local si noÃ¢â‚¬â€˜wait infaisable

# Journalisation des attentes Ã‚Â« obligatoires Ã‚Â» utilisÃƒÂ©es en JIT
JIT_WAIT_CLIENTS: Set[int] = set()  # ensemble des clients oÃƒÂ¹ une attente a rÃƒÂ©ellement ÃƒÂ©tÃƒÂ© appliquÃƒÂ©e
JIT_ACTIVATIONS: int = 0            # nb de fois oÃƒÂ¹ un fallback JIT a ÃƒÂ©tÃƒÂ© utilisÃƒÂ©
JIT_PRINT_LIMIT: int = 10           # limiter le spam en console

# ===========================
#  Utils & Data structures
# ===========================

@dataclass
class Instance:
    n: int                 # nombre de clients
    k: int                 # nombre de vÃƒÂ©hicules
    coords: np.ndarray     # (n+1, 2) : 0 = dÃƒÂ©pÃƒÂ´t, 1..n = clients
    service: np.ndarray    # (n+1,)    : service times (0 pour dÃƒÂ©pÃƒÂ´t)
    tw_a: np.ndarray       # (n+1,)    : earliest times (0 pour dÃƒÂ©pÃƒÂ´t)
    tw_b: np.ndarray       # (n+1,)    : latest times (grand intervalle pour dÃƒÂ©pÃƒÂ´t)
    dist: np.ndarray       # (n+1,n+1) : distances euclidiennes
    travel: np.ndarray     # (n+1,n+1) : temps de trajet (ici = distance)

@dataclass
class Solution:
    routes: List[List[int]]        # chaque route = liste de clients (sans dÃƒÂ©pÃƒÂ´t)
    T_ret: float                   # makespan = max retour vÃƒÂ©hicule
    D_tot: float                   # distance totale
    feasible: bool                 # faisabilitÃƒÂ© selon le mode (wait/noÃ¢â‚¬â€˜wait/hybride)
    meta: Dict[str, float]         # infos diverses (ex: best_iter, time_ms, etc.)

# ---------------------------
#   ParamÃƒÂ¨tres autoÃ¢â‚¬â€˜scalÃƒÂ©s
# ---------------------------

def auto_params(n: int) -> Dict[str, int | float]:
    """Renvoie des paramÃƒÂ¨tres adaptÃƒÂ©s ÃƒÂ  la taille n (clients)."""
    params = {}
    # ALNS iterations
    if n < 80:
        params["alns_iters"] = 12000
        params["remove_pct_min"], params["remove_pct_max"] = 6, 10
        params["k_nearest"] = 24
    elif n < 250:
        params["alns_iters"] = 60000
        params["remove_pct_min"], params["remove_pct_max"] = 8, 14
        params["k_nearest"] = 40
    elif n < 600:
        params["alns_iters"] = 120000
        params["remove_pct_min"], params["remove_pct_max"] = 10, 16
        params["k_nearest"] = 56
    else:
        params["alns_iters"] = 200000
        params["remove_pct_min"], params["remove_pct_max"] = 10, 18
        params["k_nearest"] = 64

    # Tabu settings
    if n < 80:
        params["ts_steps"] = 3000
        params["tabu_tenure_min"], params["tabu_tenure_max"] = 9, 13
        params["pr_period"] = 250
    elif n < 600:
        params["ts_steps"] = 12000
        params["tabu_tenure_min"], params["tabu_tenure_max"] = 12, 18
        params["pr_period"] = 600
    else:
        params["ts_steps"] = 20000
        params["tabu_tenure_min"], params["tabu_tenure_max"] = 14, 20
        params["pr_period"] = 800

    params["sa_T0_factor"] = 0.02   # T0 Ã¢â€°Ë† 0.02 * T
    params["sa_decay"] = 0.9995
    return params

# ===========================
#  Instance generation
# ===========================

def build_matrices(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n1 = coords.shape[0]
    dist = np.zeros((n1, n1), dtype=float)
    for i in range(n1):
        di = np.linalg.norm(coords[i] - coords, axis=1)
        dist[i] = di
    travel = dist.copy()  # ici, temps = distance
    return dist, travel


def nearest_neighbor_tour(n: int, dist: np.ndarray, seed: int) -> List[int]:
    rng = random.Random(seed)
    unvisited = set(range(1, n+1))
    cur = 0
    tour = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def gen_no_wait_friendly_instance(n: int, k: int, seed: int,
                                  coord_scale: float = 100.0,
                                  service_minmax: Tuple[int, int] = (5, 15),
                                  tw_halfwidth: float | None = None) -> Instance:
    """GÃƒÂ©nÃƒÂ¨re une instance alÃƒÂ©atoire avec fenÃƒÂªtres strictes Ã‚Â« sans attente Ã‚Â»
    construites autour d'un giant tour de rÃƒÂ©fÃƒÂ©rence.
    """
    print(f"[Gen] GÃƒÂ©nÃƒÂ©ration d'une instance alÃƒÂ©atoire (n={n}, k={k}, seed={seed})Ã¢â‚¬Â¦")
    rng = np.random.default_rng(seed)

    # CoordonnÃƒÂ©es : dÃƒÂ©pÃƒÂ´t au centre, clients uniformes
    depot = np.array([[coord_scale/2, coord_scale/2]])
    clients = rng.uniform(0, coord_scale, size=(n, 2))
    coords = np.vstack([depot, clients])  # 0 = dÃƒÂ©pÃƒÂ´t

    dist, travel = build_matrices(coords)

    # Services alÃƒÂ©atoires
    s_min, s_max = service_minmax
    service = np.zeros(n+1, dtype=float)
    service[1:] = rng.integers(s_min, s_max+1, size=n)

    # Giant tour + cumul d'arrivÃƒÂ©e prÃƒÂ©vu (dÃƒÂ©part ÃƒÂ  t=0, noÃ¢â‚¬â€˜wait)
    tour = nearest_neighbor_tour(n, dist, seed)

    cumul = 0.0
    arr_time = np.zeros(n+1, dtype=float)
    cur = 0
    for j in tour:
        cumul += travel[cur, j] + service[cur]
        arr_time[j] = cumul
        cur = j
    cumul += travel[cur, 0] + service[cur]  # retour sur dÃƒÂ©pÃƒÂ´t (info pour ÃƒÂ©chelle de temps)

    # FenÃƒÂªtres autour des temps d'arrivÃƒÂ©e de rÃƒÂ©fÃƒÂ©rence
    if tw_halfwidth is None:
        tw_halfwidth = max(15.0, 0.04 * cumul)  # ~4% du tour, bornÃƒÂ© ÃƒÂ  15

    tw_a = np.zeros(n+1, dtype=float)
    tw_b = np.zeros(n+1, dtype=float)
    tw_a[0] = 0.0
    tw_b[0] = 1e9  # dÃƒÂ©pÃƒÂ´t large

    jitter = rng.uniform(-0.25, 0.25, size=n)  # petite variation
    for idx, j in enumerate(tour, start=1):
        center = arr_time[j] + jitter[idx-1] * tw_halfwidth
        tw_a[j] = max(0.0, center - tw_halfwidth)
        tw_b[j] = center + tw_halfwidth

    print(f"[Gen] Instance prÃƒÂªte. Largeur moyenne TW Ã¢â€°Ë† {2*tw_halfwidth:.1f}")
    return Instance(n=n, k=k, coords=coords, service=service,
                    tw_a=tw_a, tw_b=tw_b, dist=dist, travel=travel)

# ===========================
#  Ãƒâ€°valuation & FaisabilitÃƒÂ©
# ===========================

def eval_route_no_wait(route: List[int], inst: Instance) -> Tuple[bool, Tuple[float, float], float, float]:
    """Ãƒâ€°value une route (sans dÃƒÂ©pÃƒÂ´t) **sans attente**.
    Renvoie : feasible, (L,U) intervalle de dÃƒÂ©part, t_retour, distance.
    """
    if not route:
        return True, (0.0, 1e9), 0.0, 0.0

    seq = [0] + route + [0]
    cumul = 0.0
    L, U = -1e18, 1e18
    dist_sum = 0.0

    for p in range(1, len(seq)-1):
        i = seq[p-1]
        j = seq[p]
        cumul += inst.travel[i, j] + inst.service[i]
        L = max(L, inst.tw_a[j] - cumul)
        U = min(U, inst.tw_b[j] - cumul)
        dist_sum += inst.dist[i, j]

    # retour dÃƒÂ©pÃƒÂ´t
    i_last = seq[-2]
    cumul_ret = cumul + inst.travel[i_last, 0] + inst.service[i_last]
    dist_sum += inst.dist[i_last, 0]

    feasible = L <= U
    if not feasible:
        return False, (L, U), math.inf, dist_sum
    s_depart = max(0.0, L)  # partir au plus tÃƒÂ´t pour minimiser le retour
    t_ret = s_depart + cumul_ret
    return True, (L, U), t_ret, dist_sum


def eval_route_wait(route: List[int], inst: Instance) -> Tuple[bool, Tuple[float, float], float, float]:
    """Ãƒâ€°value une route **avec attente autorisÃƒÂ©e**. Retourne (feasible, (0, +inf), t_retour, distance)."""
    if not route:
        return True, (0.0, 1e9), 0.0, 0.0
    seq = [0] + route + [0]
    t = 0.0
    dist_sum = 0.0
    for p in range(1, len(seq)):
        i = seq[p-1]; j = seq[p]
        t = t + inst.service[i] + inst.travel[i, j]
        t = max(t, inst.tw_a[j])  # attente si en avance
        if t > inst.tw_b[j] + 1e-9:
            return False, (0.0, 1e9), math.inf, dist_sum
        dist_sum += inst.dist[i, j]
    t_ret = t
    return True, (0.0, 1e9), t_ret, dist_sum


def eval_route_wait_collect(route: List[int], inst: Instance) -> Tuple[bool, float, float, List[int]]:
    """Version wait qui **collecte** les clients oÃƒÂ¹ une attente > 0 a ÃƒÂ©tÃƒÂ© nÃƒÂ©cessaire.
    Retourne (feasible, t_ret, dist, wait_points)."""
    if not route:
        return True, 0.0, 0.0, []
    seq = [0] + route + [0]
    t = 0.0
    dist_sum = 0.0
    wait_pts: List[int] = []
    for p in range(1, len(seq)):
        i = seq[p-1]; j = seq[p]
        arrival = t + inst.service[i] + inst.travel[i, j]
        if arrival < inst.tw_a[j]:
            wait_pts.append(j)
        t = max(arrival, inst.tw_a[j])
        if t > inst.tw_b[j] + 1e-9:
            return False, math.inf, dist_sum, wait_pts
        dist_sum += inst.dist[i, j]
    return True, t, dist_sum, wait_pts


def eval_route_jit(route: List[int], inst: Instance) -> Tuple[bool, Tuple[float, float], float, float]:
    """Essayez **noÃ¢â‚¬â€˜wait** d'abord; si infaisable, retentez en **wait** et logguez les clients
    oÃƒÂ¹ une attente a vraiment ÃƒÂ©tÃƒÂ© nÃƒÂ©cessaire. N'utilise l'attente que si obligatoire."""
    global JIT_ACTIVATIONS
    feas, LU, tret, dist_sum = eval_route_no_wait(route, inst)
    if feas:
        return feas, LU, tret, dist_sum
    # fallback : try with wait
    feas2, t2, d2, waited_nodes = eval_route_wait_collect(route, inst)
    if feas2:
        JIT_ACTIVATIONS += 1
        newly_logged = 0
        for j in waited_nodes:
            if j not in JIT_WAIT_CLIENTS:
                JIT_WAIT_CLIENTS.add(j)
                newly_logged += 1
        if JIT_ACTIVATIONS <= JIT_PRINT_LIMIT and newly_logged > 0:
            print(f"[JITÃ¢â‚¬â€˜WAIT] Attente appliquÃƒÂ©e pour client(s) {sorted(waited_nodes)} (fallback local)")
        # En mode wait, pas d'intervalle [L,U] significatif Ã¢â€ â€™ renvoyer (0,+inf)
        return True, (0.0, 1e9), t2, d2
    return False, LU, tret, dist_sum


def eval_route(route: List[int], inst: Instance) -> Tuple[bool, Tuple[float, float], float, float]:
    """Wrapper : choisit la bonne ÃƒÂ©valuation selon les drapeaux globaux.
    PrioritÃƒÂ© :
      1) ALLOW_WAIT  Ã¢â€ â€™ tout en wait
      2) ALLOW_JIT_WAIT Ã¢â€ â€™ noÃ¢â‚¬â€˜wait d'abord, puis wait seulement si nÃƒÂ©cessaire
      3) sinon Ã¢â€ â€™ noÃ¢â‚¬â€˜wait strict
    """
    if ALLOW_WAIT:
        return eval_route_wait(route, inst)
    if ALLOW_JIT_WAIT:
        return eval_route_jit(route, inst)
    return eval_route_no_wait(route, inst)


def eval_solution_generic(routes: List[List[int]], inst: Instance) -> Solution:
    feasible = True
    T = 0.0
    D = 0.0
    for r in routes:
        feas, _, tret, dist_r = eval_route(r, inst)
        if not feas:
            feasible = False
        T = max(T, tret)
        D += dist_r
    return Solution(routes=routes, T_ret=T, D_tot=D, feasible=feasible, meta={})

# ===========================
#  Construction initiale
# ===========================

def best_feasible_insertion(routes: List[List[int]], client: int, inst: Instance) -> Optional[Tuple[int, int, float]]:
    """Trouve (route_idx, pos, delta_T) pour insÃƒÂ©rer 'client'.
    Renvoie None si aucune insertion faisable.
    Tente noÃ¢â‚¬â€˜wait puis fallback JIT si activÃƒÂ©.
    """
    best = None
    best_deltaT = math.inf
    base = eval_solution_generic(routes, inst)
    base_T = base.T_ret

    for ridx, r in enumerate(routes):
        for pos in range(len(r)+1):
            new_r = r[:pos] + [client] + r[pos:]
            feas, _, _, _ = eval_route(new_r, inst)
            if not feas:
                continue
            # recompute T delta en ne changeant que la route ridx
            T_prime = 0.0
            for j, rr in enumerate(routes):
                if j == ridx:
                    feas2, _, t_ret2, _ = eval_route(new_r, inst)
                    if not feas2:
                        T_prime = math.inf
                        break
                    T_prime = max(T_prime, t_ret2)
                else:
                    feas2, _, t_ret2, _ = eval_route(rr, inst)
                    if not feas2:
                        T_prime = math.inf
                        break
                    T_prime = max(T_prime, t_ret2)
            deltaT = T_prime - base_T
            if deltaT < best_deltaT:
                best_deltaT = deltaT
                best = (ridx, pos, deltaT)
    return best


def greedy_init(inst: Instance, seed: int) -> Solution:
    """Construction gloutonne faisable (mode courant : noÃ¢â‚¬â€˜wait / wait / JIT)."""
    mode_label = "wait" if ALLOW_WAIT else ("JITÃ¢â‚¬â€˜wait" if ALLOW_JIT_WAIT else "noÃ¢â‚¬â€˜wait")
    print(f"[Init] Construction gloutonne faisable ({mode_label})Ã¢â‚¬Â¦")
    rng = random.Random(seed)
    routes = [[] for _ in range(inst.k)]
    unassigned = list(range(1, inst.n+1))
    rng.shuffle(unassigned)

    for c in unassigned:
        ins = best_feasible_insertion(routes, c, inst)
        if ins is None:
            # fallback : essayer toutes les positions avec JIT forcÃƒÂ© en dernier recours
            saved_flag = ALLOW_JIT_WAIT
            try:
                globals()["ALLOW_JIT_WAIT"] = True
                best_local = None
                best_local_T = math.inf
                for ridx in range(len(routes)):
                    r = routes[ridx]
                    for pos in range(len(r)+1):
                        new_r = r[:pos] + [c] + r[pos:]
                        feas, _, t_ret, _ = eval_route(new_r, inst)
                        if feas and t_ret < best_local_T:
                            best_local_T = t_ret
                            best_local = (ridx, pos)
                if best_local is not None:
                    ridx, pos = best_local
                    routes[ridx] = routes[ridx][:pos] + [c] + routes[ridx][pos:]
                    print(f"[Init] JITÃ¢â‚¬â€˜WAIT nÃƒÂ©cessaire pour insÃƒÂ©rer le client {c}.")
                    continue
            finally:
                globals()["ALLOW_JIT_WAIT"] = saved_flag
            # si vraiment impossible, on abandonne
            print(f"[Init] Ã¢Å¡Â Ã¯Â¸Â impossible d'insÃƒÂ©rer le client {c} mÃƒÂªme avec JITÃ¢â‚¬â€˜WAIT.")
            return Solution(routes=routes, T_ret=math.inf, D_tot=math.inf, feasible=False, meta={})
        else:
            ridx, pos, _ = ins
            r = routes[ridx]
            routes[ridx] = r[:pos] + [c] + r[pos:]

    sol = eval_solution_generic(routes, inst)
    print(f"[Init] Solution initiale : T={sol.T_ret:.2f}, D={sol.D_tot:.2f}, feasible={sol.feasible}")
    return sol

# ===========================
#  Helpers progression (loader)
# ===========================

def _progress_start(total: int, desc: str):
    """CrÃƒÂ©e une barre tqdm si dispo, sinon None (fallback console)."""
    if _HAS_TQDM:
        return tqdm(total=total, desc=desc, ncols=80, leave=False, mininterval=0.2)
    return None

# ===========================
#  ALNS Ã¢â‚¬â€ Phase A (min T)
# ===========================

def shaw_relatedness(i: int, j: int, inst: Instance, alpha_d=1.0, alpha_t=1.0) -> float:
    # SimilaritÃƒÂ© proximitÃƒÂ© gÃƒÂ©o + proximitÃƒÂ© center TW
    d = inst.dist[i, j]
    ci = 0.5 * (inst.tw_a[i] + inst.tw_b[i])
    cj = 0.5 * (inst.tw_a[j] + inst.tw_b[j])
    return alpha_d * d + alpha_t * abs(ci - cj)


def alns_phase_A_minT(inst: Instance, seed: int, params: Dict[str, float | int], verbose=True) -> Solution:
    rng = random.Random(seed)
    cur = greedy_init(inst, seed)
    best = cur

    if not cur.feasible:
        print("[ALNS] Ã¢Å¡Â Ã¯Â¸Â Solution initiale infaisable Ã¢â‚¬â€ ajustez la gÃƒÂ©nÃƒÂ©ration TW ou k.")
        return cur

    iters = int(params["alns_iters"])  
    k_nearest = int(params["k_nearest"]) 
    remove_min = int(params["remove_pct_min"]) 
    remove_max = int(params["remove_pct_max"]) 

    # PrÃƒÂ©Ã¢â‚¬â€˜calcul k-nearest (rÃƒÂ©servÃƒÂ©, pas indispensable ici)
    _ = [list(np.argsort(inst.dist[i])[:k_nearest]) for i in range(inst.n+1)]

    # SA acceptance
    T0 = max(1.0, params["sa_T0_factor"] * max(1.0, cur.T_ret))
    temp = T0
    decay = float(params["sa_decay"]) 

    print(f"[ALNS] DÃƒÂ©marrage: iters={iters}, remove%={remove_min}Ã¢â‚¬â€œ{remove_max}, k-nearest={k_nearest}, T0Ã¢â€°Ë†{T0:.3f}")

    pbar = _progress_start(iters, "ALNS Phase A")
    t_start = time.time()

    def destroy_random(routes: List[List[int]]):
        allc = [c for r in routes for c in r]
        if not allc:
            return [list(r) for r in routes], []
        q = max(1, len(allc) * rng.randrange(remove_min, remove_max+1) // 100)
        removed = rng.sample(allc, q)
        new_routes = [[c for c in r if c not in removed] for r in routes]
        return new_routes, removed

    def destroy_shaw(routes: List[List[int]]):
        allc = [c for r in routes for c in r]
        if not allc:
            return [list(r) for r in routes], []
        seedc = rng.choice(allc)
        q = max(1, len(allc) * rng.randrange(remove_min, remove_max+1) // 100)
        sim = [(c, shaw_relatedness(seedc, c, inst)) for c in allc]
        sim.sort(key=lambda x: x[1])
        removed = [c for c,_ in sim[:q]]
        new_routes = [[c for c in r if c not in removed] for r in routes]
        return new_routes, removed

    def repair_best_insertion(routes: List[List[int]], removed: List[int]) -> Optional[List[List[int]]]:
        rr = [list(r) for r in routes]
        for c in removed:
            placed = False
            cand_routes = sorted(range(inst.k), key=lambda ridx: len(rr[ridx]))
            for ridx in cand_routes:
                r = rr[ridx]
                best_pos = None
                best_delta = math.inf
                base_eval = eval_solution_generic(rr, inst)
                base_T = base_eval.T_ret
                for pos in range(len(r)+1):
                    new_r = r[:pos] + [c] + r[pos:]
                    feas, _, _, _ = eval_route(new_r, inst)
                    if not feas:
                        continue
                    # delta T local
                    T_prime = 0.0
                    for j, rx in enumerate(rr):
                        if j == ridx:
                            feas2, _, t2, _ = eval_route(new_r, inst)
                            if not feas2:
                                T_prime = math.inf; break
                            T_prime = max(T_prime, t2)
                        else:
                            feas2, _, t2, _ = eval_route(rx, inst)
                            if not feas2:
                                T_prime = math.inf; break
                            T_prime = max(T_prime, t2)
                    delta = T_prime - base_T
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                if best_pos is not None:
                    rr[ridx] = r[:best_pos] + [c] + r[best_pos:]
                    placed = True
                    break
            if not placed:
                return None
        return rr

    for it in range(1, iters+1):
        # choisir un destroy
        if rng.random() < 0.5:
            routes_d, removed = destroy_random(best.routes)
        else:
            routes_d, removed = destroy_shaw(best.routes)

        # repair
        routes_r = repair_best_insertion(routes_d, removed)
        if routes_r is None:
            temp *= decay
            # update loader
            if _HAS_TQDM and pbar is not None:
                pbar.set_postfix(T=f"{cur.T_ret:.2f}", best=f"{best.T_ret:.2f}")
                pbar.update(1)
            else:
                if verbose and (it == 1 or it % max(1, iters//100) == 0):
                    elapsed = time.time() - t_start
                    pct = 100.0 * it / iters
                    eta = elapsed * (iters/it - 1.0)
                    sys.stdout.write(f"\r[ALNS] {pct:5.1f}%  T={cur.T_ret:.2f} best={best.T_ret:.2f}  ETA~{eta:5.1f}s")
                    sys.stdout.flush()
            continue

        cand = eval_solution_generic(routes_r, inst)
        dE = cand.T_ret - cur.T_ret
        accept = (dE <= 0) or (math.exp(-dE / max(1e-9, temp)) > rng.random())
        if accept:
            cur = cand
            if cand.T_ret < best.T_ret - 1e-9 or (abs(cand.T_ret - best.T_ret) <= 1e-9 and cand.D_tot < best.D_tot):
                best = cand
                if verbose and not _HAS_TQDM:
                    print(f"\n[ALNS] Ã¢Å“â€œ it={it}: nouveau best T={best.T_ret:.2f} D={best.D_tot:.2f}")
        temp *= decay

        # update loader
        if _HAS_TQDM and pbar is not None:
            pbar.set_postfix(T=f"{cur.T_ret:.2f}", best=f"{best.T_ret:.2f}")
            pbar.update(1)
        else:
            if verbose and (it == 1 or it % max(1, iters//100) == 0):
                elapsed = time.time() - t_start
                pct = 100.0 * it / iters
                eta = elapsed * (iters/it - 1.0)
                sys.stdout.write(f"\r[ALNS] {pct:5.1f}%  T={cur.T_ret:.2f} best={best.T_ret:.2f}  ETA~{eta:5.1f}s")
                sys.stdout.flush()

    if _HAS_TQDM and pbar is not None:
        pbar.close()
    else:
        print()  # retour ligne pour le loader console

    best.meta.update({"phase": "A", "iters": iters})
    print(f"[ALNS] TerminÃƒÂ©. Best T={best.T_ret:.2f}, D={best.D_tot:.2f}")
    return best

# ===========================
#  Tabu Ã¢â‚¬â€ Phase B (min D | TÃ¢â€°Â¤T*)
# ===========================

def tabu_phase_B_minD(inst: Instance, solA: Solution, params: Dict[str, float | int], seed: int, verbose=True) -> Solution:
    rng = random.Random(seed)
    T_star = solA.T_ret
    cur = Solution(routes=[list(r) for r in solA.routes], T_ret=solA.T_ret, D_tot=solA.D_tot, feasible=solA.feasible, meta={})
    best = cur

    steps = int(params["ts_steps"]) 
    tenure_min = int(params["tabu_tenure_min"]) 
    tenure_max = int(params["tabu_tenure_max"]) 

    # Tabu list: key=(client, prev)
    tabu: Dict[Tuple[int,int], int] = {}

    def is_tabu(move_key: Tuple[int,int], iter_idx: int) -> bool:
        return tabu.get(move_key, -1) > iter_idx

    def add_tabu(move_key: Tuple[int,int], iter_idx: int):
        tenure = rng.randint(tenure_min, tenure_max)
        tabu[move_key] = iter_idx + tenure

    def evaluate_routes_T(routes: List[List[int]]) -> Tuple[bool, float, float]:
        T = 0.0; D = 0.0; feas_all = True
        for r in routes:
            feas, _, tret, dist_r = eval_route(r, inst)
            if not feas:
                feas_all = False
            T = max(T, tret)
            D += dist_r
        return feas_all, T, D

    print(f"[Tabu] DÃƒÂ©marrage polishing distance avec contrainte T Ã¢â€°Â¤ {T_star:.2f}Ã¢â‚¬Â¦")

    pbar = _progress_start(steps, "Tabu Phase B")
    t_start = time.time()

    for it in range(1, steps+1):
        best_nbr = None
        best_D = math.inf
        best_move = None

        # GÃƒÂ©nÃƒÂ¨re un petit voisinage : relocate 1-client, swap 1Ã¢â‚¬â€œ1, 2-opt* inter-route
        routes = cur.routes
        k_routes = len(routes)

        # Relocate 1-client
        for ra in range(k_routes):
            rA = routes[ra]
            for pa in range(len(rA)):
                c = rA[pa]
                prev = rA[pa-1] if pa > 0 else 0
                for rb in range(k_routes):
                    rB = routes[rb]
                    for pb in range(len(rB)+1):
                        if ra == rb and (pb == pa or pb == pa+1):
                            continue
                        new_routes = [list(r) for r in routes]
                        new_routes[ra] = rA[:pa] + rA[pa+1:]
                        new_routes[rb] = rB[:pb] + [c] + rB[pb:]

                        feas, T, D = evaluate_routes_T(new_routes)
                        if (not feas) or T > T_star + 1e-9:
                            continue
                        key = (c, prev)
                        if is_tabu(key, it) and D >= best.D_tot - 1e-9:
                            continue
                        if D < best_D:
                            best_D = D
                            best_nbr = new_routes
                            best_move = ("relocate", key)

        # Swap 1Ã¢â‚¬â€œ1
        for ra in range(k_routes):
            rA = routes[ra]
            for rb in range(ra+1, k_routes):
                rB = routes[rb]
                for pa in range(len(rA)):
                    for pb in range(len(rB)):
                        ca, cb = rA[pa], rB[pb]
                        prev_a = rA[pa-1] if pa > 0 else 0
                        prev_b = rB[pb-1] if pb > 0 else 0
                        new_routes = [list(r) for r in routes]
                        new_routes[ra][pa] = cb
                        new_routes[rb][pb] = ca
                        feas, T, D = evaluate_routes_T(new_routes)
                        if (not feas) or T > T_star + 1e-9:
                            continue
                        tabu_block = (is_tabu((ca, prev_a), it) or is_tabu((cb, prev_b), it))
                        if tabu_block and D >= best.D_tot - 1e-9:
                            continue
                        if D < best_D:
                            best_D = D
                            best_nbr = new_routes
                            best_move = ("swap", (ca, prev_a, cb, prev_b))

        # 2-opt* inter-route (ÃƒÂ©changes de suffixes)
        for ra in range(k_routes):
            rA = routes[ra]
            for rb in range(ra+1, k_routes):
                rB = routes[rb]
                for cutA in range(1, len(rA)+1):
                    for cutB in range(1, len(rB)+1):
                        new_routes = [list(r) for r in routes]
                        new_routes[ra] = rA[:cutA] + rB[cutB:]
                        new_routes[rb] = rB[:cutB] + rA[cutA:]
                        feas, T, D = evaluate_routes_T(new_routes)
                        if (not feas) or T > T_star + 1e-9:
                            continue
                        if D < best_D:
                            best_D = D
                            best_nbr = new_routes
                            best_move = ("2opt*", None)

        if best_nbr is None:
            # update loader
            if _HAS_TQDM and pbar is not None:
                pbar.set_postfix(D=f"{cur.D_tot:.2f}", best=f"{best.D_tot:.2f}")
                pbar.update(1)
            else:
                if verbose and (it == 1 or it % max(1, steps//100) == 0):
                    elapsed = time.time() - t_start
                    pct = 100.0 * it / steps
                    eta = elapsed * (steps/it - 1.0)
                    sys.stdout.write(f"\r[Tabu] {pct:5.1f}%  D={cur.D_tot:.2f} best={best.D_tot:.2f}  ETA~{eta:5.1f}s")
                    sys.stdout.flush()
            continue

        # Appliquer meilleur voisin
        cur = eval_solution_generic(best_nbr, inst)
        # Mettre ÃƒÂ  jour TABU
        if best_move[0] == "relocate":
            add_tabu(best_move[1], it)
        elif best_move[0] == "swap":
            _, (ca, prev_a, cb, prev_b) = best_move
            add_tabu((ca, prev_a), it)
            add_tabu((cb, prev_b), it)

        if cur.D_tot + 1e-9 < best.D_tot:
            best = cur
            if verbose and not _HAS_TQDM:
                print(f"\n[Tabu] Ã¢Å“â€œ it={it}: nouveau best D={best.D_tot:.2f} (T={best.T_ret:.2f})")

        # update loader
        if _HAS_TQDM and pbar is not None:
            pbar.set_postfix(D=f"{cur.D_tot:.2f}", best=f"{best.D_tot:.2f}")
            pbar.update(1)
        else:
            if verbose and (it == 1 or it % max(1, steps//100) == 0):
                elapsed = time.time() - t_start
                pct = 100.0 * it / steps
                eta = elapsed * (steps/it - 1.0)
                sys.stdout.write(f"\r[Tabu] {pct:5.1f}%  D={cur.D_tot:.2f} best={best.D_tot:.2f}  ETA~{eta:5.1f}s")
                sys.stdout.flush()

    if _HAS_TQDM and pbar is not None:
        pbar.close()
    else:
        print()

    best.meta.update({"phase": "B", "steps": steps, "T_star": T_star})
    print(f"[Tabu] TerminÃƒÂ©. D*={best.D_tot:.2f} sous TÃ¢â€°Â¤{T_star:.2f}")
    return best

# ===========================
#  Visualisation (plot + anim)
# ===========================

def plot_solution(inst: Instance, sol: Solution, title: str = "Solution"):
    colors = plt.cm.get_cmap('tab20', len(sol.routes))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(inst.coords[1:,0], inst.coords[1:,1], s=30, label="Clients")
    ax.scatter(inst.coords[0,0], inst.coords[0,1], s=120, marker='*', label="DÃƒÂ©pÃƒÂ´t")
    for idx, r in enumerate(sol.routes):
        x = [inst.coords[0,0]] + [inst.coords[i,0] for i in r] + [inst.coords[0,0]]
        y = [inst.coords[0,1]] + [inst.coords[i,1] for i in r] + [inst.coords[0,1]]
        ax.plot(x, y, '-', alpha=0.8, label=f"Route {idx+1}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()


def animate_solution(inst: Instance, sol: Solution, interval_ms: int = 60):
    # PrÃƒÂ©pare les trajectoires et les temps
    routes = sol.routes
    paths = []  # liste de (points_xy, times)
    T_max = 0.0
    for r in routes:
        seq = [0] + r + [0]
        pts = [inst.coords[i] for i in seq]
        times = [0.0]
        if ALLOW_WAIT:
            t = 0.0
            for p in range(1, len(seq)):
                i = seq[p-1]; j = seq[p]
                t = t + inst.service[i] + inst.travel[i, j]
                t = max(t, inst.tw_a[j])  # attente possible
                times.append(t)
        else:
            feas, (L,U), _, _ = eval_route_no_wait(r, inst)
            s_dep = max(0.0, L)
            t = s_dep
            for p in range(1, len(seq)):
                i = seq[p-1]; j = seq[p]
                t += inst.travel[i, j] + inst.service[i]
                times.append(t)
        paths.append((pts, times))
        T_max = max(T_max, times[-1])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(inst.coords[1:,0], inst.coords[1:,1], s=30, label="Clients")
    ax.scatter(inst.coords[0,0], inst.coords[0,1], s=120, marker='*', label="DÃƒÂ©pÃƒÂ´t")
    ax.legend(loc='upper right')
    ax.grid(True)

    # Traces statiques des routes
    for r in sol.routes:
        seq = [0] + r + [0]
        x = [inst.coords[i,0] for i in seq]
        y = [inst.coords[i,1] for i in seq]
        ax.plot(x, y, lw=0.8, alpha=0.3)

    # marqueurs mobiles par vÃƒÂ©hicule
    artists = []
    for _ in routes:
        (pt,) = ax.plot([], [], 'o', ms=8)
        artists.append(pt)

    def interp(p0, p1, t0, t1, t):
        if t1 == t0:
            return p1
        lam = (t - t0) / (t1 - t0)
        lam = min(1.0, max(0.0, lam))
        return p0 * (1-lam) + p1 * lam

    def init():
        for a in artists:
            a.set_data([], [])
        return artists

    def update(frame_t):
        t = frame_t
        for v, (pts, times) in enumerate(paths):
            # trouver segment courant
            idx = 0
            while idx < len(times)-1 and times[idx+1] < t:
                idx += 1
            if idx >= len(times)-1:
                x, y = pts[-1]
            else:
                p0 = pts[idx]
                p1 = pts[idx+1]
                t0 = times[idx]
                t1 = times[idx+1]
                p = interp(p0, p1, t0, t1, t)
                x, y = p
            artists[v].set_data(x, y)
        return artists

    frames = max(1, int(T_max))
    _ = animation.FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval_ms, blit=True, repeat=False)
    plt.show()

# ===========================
#  Runner principal
# ===========================

def run_solve(n_clients: int, k_trucks: int, seed: int,
              allow_wait: bool = False,
              jit_wait_if_needed: bool = True,
              time_budget_s: Optional[float] = None,
              verbose: bool = True,
              animate: bool = True) -> Dict:
    t0 = time.time()

    # bascule des modes globaux
    global ALLOW_WAIT, ALLOW_JIT_WAIT, JIT_WAIT_CLIENTS, JIT_ACTIVATIONS
    ALLOW_WAIT = allow_wait
    ALLOW_JIT_WAIT = (not allow_wait) and jit_wait_if_needed
    JIT_WAIT_CLIENTS.clear(); JIT_ACTIVATIONS = 0

    mode_str = (
        "attente autorisÃƒÂ©e" if ALLOW_WAIT else (
        "sans attente (JITÃ¢â‚¬â€˜WAIT activÃƒÂ© si nÃƒÂ©cessaire)" if ALLOW_JIT_WAIT else
        "sans attente (strict)"))
    print(f"[Run] Mode : {mode_str}")

    inst = gen_no_wait_friendly_instance(n_clients, k_trucks, seed)
    params = auto_params(n_clients)

    print("[Run] Phase A : recherche d'une solution avec makespan minimalÃ¢â‚¬Â¦")
    solA = alns_phase_A_minT(inst, seed=seed, params=params, verbose=verbose)

    if not solA.feasible or math.isinf(solA.T_ret):
        print("[Run] Ã¢Å“â€” Ãƒâ€°chec Phase A (infaisable). ArrÃƒÂªt.")
        return {
            "n": n_clients, "k": k_trucks, "seed": seed,
            "feasible": False
        }

    print("[Run] Phase B : optimisation distance sous T Ã¢â€°Â¤ T*Ã¢â‚¬Â¦")
    solB = tabu_phase_B_minD(inst, solA, params=params, seed=seed+1, verbose=verbose)

    tf = time.time()
    elapsed = (tf - t0) * 1000.0

    # Pas de gap VRPLIB pour les instances alÃƒÂ©atoires Ã¢â€ â€™ N/A
    print("[Run] Ã¢Å“â€¦ Solution finale trouvÃƒÂ©e.")
    print(f"      Ã¢â€ â€™ Clients={n_clients}, Camions={k_trucks}, Seed={seed}")
    print(f"      Ã¢â€ â€™ Makespan T*={solB.T_ret:.2f}")
    print(f"      Ã¢â€ â€™ Distance D={solB.D_tot:.2f}")
    print(f"      Ã¢â€ â€™ Gap = N/A (instances alÃƒÂ©atoires)")
    if JIT_WAIT_CLIENTS:
        print(f"      Ã¢â€ â€™ JITÃ¢â‚¬â€˜WAIT appliquÃƒÂ© pour client(s) : {sorted(JIT_WAIT_CLIENTS)} (fallback local uniquement)")
    print()

    plot_solution(inst, solB, title=f"VRPTW {'wait' if ALLOW_WAIT else ('JIT-wait' if ALLOW_JIT_WAIT else 'no-wait')} - T*={solB.T_ret:.1f}, D={solB.D_tot:.1f}")
    if animate:
        print("[Run] Animation des tournÃƒÂ©es (camions en mouvement)Ã¢â‚¬Â¦")
        animate_solution(inst, solB)

    return {
        "n": n_clients,
        "k": k_trucks,
        "seed": seed,
        "T": solB.T_ret,
        "D": solB.D_tot,
        "gap": None,
        "routes": solB.routes,
        "elapsed_ms": elapsed,
        "jit_wait_clients": sorted(JIT_WAIT_CLIENTS),
    }

# ===========================
#  Tests rapides (non interactifs)
# ===========================

def _self_test():
    print("\n=== SELF-TEST: dÃƒÂ©marrage ===")
    global ALLOW_WAIT, ALLOW_JIT_WAIT

    # 1) Petit no-wait strict
    ALLOW_WAIT, ALLOW_JIT_WAIT = False, False
    inst = gen_no_wait_friendly_instance(12, 3, 1)
    solA = alns_phase_A_minT(inst, seed=1, params=auto_params(inst.n), verbose=False)
    assert solA.feasible and math.isfinite(solA.T_ret)
    solB = tabu_phase_B_minD(inst, solA, params=auto_params(inst.n), seed=2, verbose=False)
    assert solB.feasible and math.isfinite(solB.D_tot)
    print("[TEST] no-wait strict OK Ã¢â‚¬â€ T*={:.2f}, D={:.2f}".format(solB.T_ret, solB.D_tot))

    # 2) Mode wait
    ALLOW_WAIT, ALLOW_JIT_WAIT = True, False
    inst = gen_no_wait_friendly_instance(12, 3, 2)
    solA = alns_phase_A_minT(inst, seed=2, params=auto_params(inst.n), verbose=False)
    assert solA.feasible and math.isfinite(solA.T_ret)
    solB = tabu_phase_B_minD(inst, solA, params=auto_params(inst.n), seed=3, verbose=False)
    assert solB.feasible and math.isfinite(solB.D_tot)
    print("[TEST] wait OK Ã¢â‚¬â€ T*={:.2f}, D={:.2f}".format(solB.T_ret, solB.D_tot))

    # 3) JITÃ¢â‚¬â€˜WAIT (fallback local)
    ALLOW_WAIT, ALLOW_JIT_WAIT = False, True
    inst = gen_no_wait_friendly_instance(50, 5, 3)
    solA = alns_phase_A_minT(inst, seed=3, params=auto_params(inst.n), verbose=False)
    solB = tabu_phase_B_minD(inst, solA, params=auto_params(inst.n), seed=4, verbose=False)
    assert solB.feasible
    print("[TEST] JITÃ¢â‚¬â€˜WAIT OK Ã¢â‚¬â€ T*={:.2f}, D={:.2f}".format(solB.T_ret, solB.D_tot))

    print("=== SELF-TEST: OK ===\n")

# ===========================
#  Exemple d'utilisation
# ===========================
if __name__ == "__main__":
    # Lancer tests: python script.py --self-test
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)

    try:
        print("=== Solveur VRPTW (no-wait / wait / JIT-wait) - ALNS + Tabu ===")
        k = int(input("Nombre de camions k : "))
        n = int(input("Nombre de clients n (10..2000) : "))
        seed = int(input("Seed de generation : "))
        jit_in = input("Activer le JIT-WAIT (attente seulement si obligatoire) ? (o/n) : ").strip().lower()
        allow_wait = False
        jit_wait_if_needed = jit_in in ("o", "oui", "y", "yes")
    except Exception:
        print("EntrÃƒÂ©es invalides, on prend k=5, n=100, seed=42 par dÃƒÂ©faut.")
        k, n, seed = 5, 100, 42
        allow_wait = False
        jit_wait_if_needed = True

    _ = run_solve(n_clients=n, k_trucks=k, seed=seed, allow_wait=allow_wait, jit_wait_if_needed=jit_wait_if_needed, animate=True)

