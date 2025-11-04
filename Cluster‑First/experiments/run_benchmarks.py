import os
import time
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt

from solver.data import load_vrplib
from solver.solver import solve_vrp

# Dossier data relatif à ce script
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Instances locales (tu peux modifier/ajouter ici)
INSTANCES = [
    os.path.join(DATA_DIR, "A-n32-k5.vrp"),
    os.path.join(DATA_DIR, "X-n101-k25.vrp"),
    # Provisoire pour ~200 clients en attendant M-n200-k17 :
    os.path.join(DATA_DIR, "cvrplib", "Vrp-Set-X", "X", "X-n200-k36.vrp"),
]

def read_opt_cost_local(path_vrp: str):
    """
    Tente de lire le coût optimal depuis un fichier .sol à côté du .vrp.
    1) via vrplib.read_solution si dispo
    2) fallback: extraction naïve d'un nombre depuis le .sol
    """
    path_sol = os.path.splitext(path_vrp)[0] + ".sol"
    if not os.path.exists(path_sol):
        return None

    # 1) tentative vrplib
    try:
        import vrplib  # type: ignore
        sol = vrplib.read_solution(path_sol)
        if isinstance(sol, dict) and "cost" in sol:
            return float(sol["cost"])
    except Exception:
        pass

    # 2) extraction naïve
    try:
        with open(path_sol, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        # récupère tous les tokens qui ressemblent à des nombres
        nums = []
        for tok in txt.replace(",", " ").split():
            t = tok.strip()
            if t.replace(".", "", 1).isdigit():
                try:
                    nums.append(float(t))
                except Exception:
                    pass
        return min(nums) if nums else None
    except Exception:
        return None

def run_experiment(path_vrp: str, runs: int = 20):
    inst = load_vrplib(path_vrp)
    opt_cost = read_opt_cost_local(path_vrp)

    costs, times, makespans = [], [], []
    feas_count = 0

    for r in range(runs):
        t0 = time.perf_counter()
        res = solve_vrp(inst, rng_seed=r, tabu_max_iter=1500, tabu_no_improve=150)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        feas_count += int(res["feasible"])
        costs.append(res["cost"])
        mksp = float(res.get("makespan", 0.0))
        makespans.append(mksp)

        print(
            f"    Run {r + 1:02d}/{runs}: duration={elapsed:.2f}s · dernier camion au dépôt={mksp:.2f}"
        )

    avg_cost = mean(costs)
    std_cost = stdev(costs) if len(costs) > 1 else 0.0
    avg_time = mean(times)
    avg_makespan = mean(makespans) if makespans else 0.0

    print(f"\nInstance {os.path.basename(path_vrp)}")
    print(f"  Feasible runs: {feas_count}/{runs}")
    print(f"  Cost: mean={avg_cost:.2f}  std={std_cost:.2f}")
    print(f"  Time: mean={avg_time:.2f}s")
    print(f"  Makespan (dernier camion): mean={avg_makespan:.2f}")

    if opt_cost is not None:
        gaps = [100.0 * (c - opt_cost) / opt_cost for c in costs]
        print(f"  GAP mean: {mean(gaps):.2f}%")

    # Boxplot des coûts
    plt.figure()
    plt.boxplot(costs)
    plt.title(f"Boxplot des coûts — {os.path.basename(path_vrp)}")
    plt.ylabel("Coût total")
    plt.show()

    # Distribution des coûts
    plt.figure()
    plt.hist(costs, bins=8)
    plt.title(f"Distribution des coûts — {os.path.basename(path_vrp)}")
    plt.xlabel("Coût")
    plt.ylabel("Fréquence")
    plt.show()

if __name__ == "__main__":
    for p in INSTANCES:
        run_experiment(p, runs=20)
