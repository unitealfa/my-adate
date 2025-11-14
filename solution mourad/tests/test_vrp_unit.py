# -*- coding: utf-8 -*-
"""
Unit tests VRP/VRPTW — jouets pédagogiques + invariants.
"""

import numpy as np
import pytest
import vrp_solver as vrp

DEBUG = True
def dbg(msg: str) -> None:
    if DEBUG:
        print(msg)

def route_load(inst, r):
    return float(sum(inst.demand[c] for c in r[1:-1]))

def debug_route(inst, r, title="route"):
    ok, arr = vrp.recompute_arrivals(inst, r)
    dbg(f"[{title}] {r} | load={route_load(inst, r):.1f} | TW_ok={ok}")
    if ok:
        dbg(f"[{title}] arrivals={arr}")

# ---------------- Fixtures jouets ----------------
def mk_inst_cvrp():
    n = 5
    coords = np.array([
        [0.0, 0.0],  # depot
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ], dtype=float)
    demand = np.array([0, 2, 2, 2, 2, 2], dtype=float)
    ready   = np.zeros(n+1, dtype=float)
    due     = np.full(n+1, 1e12, dtype=float)
    service = np.zeros(n+1, dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    return vrp.Instance(
        name="CVRP_TOY",
        n=n,
        coords=coords,
        demand=demand,
        ready=ready,
        due=due,
        service=service,
        capacity=6,
        k=4,
        dist=dist,
        has_tw=False,
    )

def mk_inst_tw():
    n = 4
    coords = np.array([
        [0.0, 0.0],  # depot
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 3.0],
        [3.0, 3.0],
    ], dtype=float)
    demand = np.array([0, 1, 1, 1, 1], dtype=float)
    ready   = np.array([0.0, 5.0, 0.0, 0.0, 0.0], dtype=float)
    due     = np.array([1e9, 100.0, 100.0, 100.0, 100.0], dtype=float)
    service = np.array([0.0, 2.0, 1.0, 0.0, 0.0], dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    return vrp.Instance(
        name="VRPTW_TOY",
        n=n,
        coords=coords,
        demand=demand,
        ready=ready,
        due=due,
        service=service,
        capacity=100,
        k=3,
        dist=dist,
        has_tw=True,
    )

# --------- 1) Capacité ---------
def test_capacity_ok_true_false():
    inst = mk_inst_cvrp()
    route_ok  = [0, 1, 2, 0]
    route_bad = [0, 1, 2, 3, 4, 0]
    debug_route(inst, route_ok,  "capacity/OK")
    debug_route(inst, route_bad, "capacity/BAD")
    assert vrp.capacity_ok(inst, route_ok)
    assert not vrp.capacity_ok(inst, route_bad)

# --------- 2) TW : attente & violation ---------
def test_recompute_arrivals_waiting_then_ok():
    inst = mk_inst_tw()
    r = [0, 1, 2, 0]
    ok, arr = vrp.recompute_arrivals(inst, r)
    debug_route(inst, r, "arrivals/wait")
    assert ok
    assert pytest.approx(arr[1], rel=1e-8) == inst.ready[1]
    depart1 = max(arr[1], inst.ready[1]) + inst.service[1]
    assert depart1 >= 7.0 - 1e-9

def test_recompute_arrivals_violation_due():
    inst = mk_inst_tw()
    r = [0, 1, 2, 0]
    ok, arr = vrp.recompute_arrivals(inst, r)
    assert ok
    eta2 = arr[2]
    inst_bad = vrp.Instance(**{**inst.__dict__, "due": inst.due.copy()})
    inst_bad.due[2] = eta2 - 0.5
    ok2, _ = vrp.recompute_arrivals(inst_bad, r)
    assert not ok2

# --------- 3) Monotonie coût ---------
def test_route_cost_monotone_when_adding_node():
    inst = mk_inst_cvrp()
    c_base = vrp.route_cost(inst.dist, np.array([0, 1, 0], dtype=np.int32))
    c_add  = vrp.route_cost(inst.dist, np.array([0, 1, 2, 0], dtype=np.int32))
    dbg(f"[monotone] cost[0-1-0]={c_base:.6f}, cost[0-1-2-0]={c_add:.6f}")
    assert c_add + 1e-9 >= c_base

# --------- 4) Opérateurs gardent la faisabilité ---------
@pytest.mark.parametrize("op_name", ["or_opt_intra", "two_opt_intra"])
def test_intra_ops_keep_feasible(op_name):
    inst = mk_inst_cvrp()
    r = [0, 1, 2, 3, 0]
    fn = getattr(vrp, op_name)
    changed, r2 = fn(inst, r)
    dbg(f"[intra:{op_name}] changed={changed}, r->{r2}")
    if changed:
        assert vrp.capacity_ok(inst, r2)
        ok, _ = vrp.recompute_arrivals(inst, r2)
        assert ok

@pytest.mark.parametrize("op_name", ["relocate_inter", "swap_11", "two_opt_star"])
def test_inter_ops_keep_feasible(op_name):
    inst = mk_inst_cvrp()
    r1 = [0, 1, 2, 0]
    r2 = [0, 3, 4, 5, 0]
    fn = getattr(vrp, op_name)
    out = fn(inst, r1, r2)
    assert isinstance(out, tuple) and len(out) == 3
    changed, nr1, nr2 = out
    dbg(f"[inter:{op_name}] changed={changed}, r1->nr1={r1}->{nr1}, r2->nr2={r2}->{nr2}")
    if changed:
        assert vrp.capacity_ok(inst, nr1)
        assert vrp.capacity_ok(inst, nr2)
        ok1, _ = vrp.recompute_arrivals(inst, nr1)
        ok2, _ = vrp.recompute_arrivals(inst, nr2)
        assert ok1 and ok2

def test_regret_repair_restores_all_customers_and_feasibility():
    inst = mk_inst_cvrp()
    routes = [[0, 1, 2, 0], [0, 3, 4, 5, 0]]
    removed = [2, 3]
    base = vrp.remove_customers(routes, removed)
    repaired = vrp.regret_repair(inst, base, removed)
    seen = [c for r in repaired for c in r[1:-1]]
    assert sorted(seen) == list(range(1, inst.n + 1))
    for r in repaired:
        assert vrp.capacity_ok(inst, r)
        ok, _ = vrp.recompute_arrivals(inst, r)
        assert ok

# --------- 5) Split CVRP/VRPTW ---------
def test_split_cvrp_produces_feasible_routes():
    inst = mk_inst_cvrp()
    tour = [1, 2, 3, 4, 5]
    routes = vrp.split_cvrp(inst, tour)
    dbg(f"[split_cvrp] routes={routes}")
    assert routes is not None and len(routes) >= 1
    seen = [c for r in routes for c in r[1:-1]]
    assert sorted(seen) == tour
    for r in routes:
        assert vrp.capacity_ok(inst, r)
        ok, _ = vrp.recompute_arrivals(inst, r)
        assert ok

def test_split_vrptw_feasible_and_infeasible():
    inst = mk_inst_tw()
    tour = [1, 2, 3, 4]
    r_ok = vrp.split_vrptw(inst, tour)
    dbg(f"[split_vrptw] feasible routes={r_ok}")
    assert r_ok is not None

    inst_bad = vrp.Instance(**{**inst.__dict__, "due": inst.due.copy()})
    earliest2 = float(inst.dist[0, 2])  # ready[2]=0 => earliest arrival at 2
    inst_bad.due[2] = max(0.0, earliest2 - 1e-3)  # STRICT < earliest
    r_bad = vrp.split_vrptw(inst_bad, tour)
    dbg(f"[split_vrptw] infeasible due2={inst_bad.due[2]:.6f} < earliest2={earliest2:.6f} -> {r_bad}")
    assert r_bad is None

# --------- 6) validate_solution ---------
def test_validate_solution_passes_on_feasible_routes():
    inst = mk_inst_cvrp()
    routes = [[0, 1, 2, 0], [0, 3, 4, 5, 0]]
    vrp.validate_solution(inst, routes)

def test_validate_solution_raises_on_capacity_violation():
    inst = mk_inst_cvrp()
    bad = [[0, 1, 2, 3, 4, 0]]
    with pytest.raises(ValueError):
        vrp.validate_solution(inst, bad)
