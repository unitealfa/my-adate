import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.data import load_vrplib
from solver.solver import solve_vrp
from solver.evaluator import eval_route

def quick_check(path_vrp):
    inst = load_vrplib(path_vrp)
    res = solve_vrp(inst, rng_seed=42, tabu_max_iter=500, tabu_no_improve=100)
    assert res["feasible"], "Solution infaisable"
    for r, k in zip(res["routes"], res["veh_types"]):
        er = eval_route(inst, r, k)
        assert er.capacity_excess == 0.0 and er.tw_violation == 0.0
    required = inst.num_veh_by_type[0]
    if required < 1e8:
        assert len(res["routes"]) >= int(round(required)), "Tous les camions disponibles doivent être utilisés"
        assert all(len(route) > 0 for route in res["routes"]), "Aucun camion ne doit rester vide"
    print("OK:", os.path.basename(path_vrp), "| veh:", res["used_vehicles"], "| cost:", f"{res['cost']:.2f}")

if __name__ == "__main__":
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    quick_check(os.path.join(base, "A-n32-k5.vrp"))
    quick_check(os.path.join(base, "X-n101-k25.vrp"))
