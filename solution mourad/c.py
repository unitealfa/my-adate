import importlib, os, re, time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 🔧 CONFIGURATION
# ============================================================
SOLVER_MODULE = "vrp_solver"  # nom de ton fichier principal sans .py
INSTANCE_PATH = Path("data/cvrplib/Vrp-Set-Solomon/C101.txt")  # chemin instance
OUTPUT_CSV = Path("convergence_C101.csv")
OUTPUT_PNG = Path("convergence_C101.png")
LOOPS = 500   # nb boucles max pour voir la convergence
NNK = 20
# ============================================================

# --- Import dynamique du solver ---
solver = importlib.import_module(SOLVER_MODULE)

# On va injecter un "logger" pour enregistrer bestc à chaque itération
original_hgs_solve = solver.hgs_solve

best_cost_log = []

def patched_hgs_solve(inst, time_loops, pop_size, **kwargs):
    """Wrapper de hgs_solve pour logger le meilleur coût à chaque itération"""
    global best_cost_log
    best_cost_log = []

    def fit(rs): return solver.total_cost(inst, rs)

    # On copie la boucle principale de ton hgs_solve, mais on y ajoute un log
    pop_tours = []
    pop_routes = []
    nn = solver.build_nn(inst.dist, kwargs.get("nnk", 20)) if kwargs.get("fast", True) else None

    # Initialisation population
    for _ in range(min(pop_size, 8)):
        routes = solver.build_seed(inst) if inst.has_tw else solver.build_seed_sweep(inst)
        routes = solver.rvnd(inst, routes, max_loops=3, nn=nn)
        pop_routes.append(routes)
        pop_tours.append(solver.tour_from_routes(routes))
    nodes = list(range(1, inst.n+1))
    for _ in range(max(0, pop_size - len(pop_tours))):
        import random
        random.shuffle(nodes)
        tour = nodes.copy()
        routes = (solver.split_vrptw(inst, tour) if inst.has_tw else solver.split_cvrp(inst, tour)) or [[0,u,0] for u in tour]
        routes = solver.rvnd(inst, routes, max_loops=2, nn=nn)
        pop_routes.append(routes)
        pop_tours.append(solver.tour_from_routes(routes))

    best = min(pop_routes, key=fit)
    bestc = fit(best)
    best_cost_log.append(bestc)  # 🔹 log du coût initial

    import numpy as np, random
    t_start = time.perf_counter()
    for t in range(time_loops):
        # --- génération simple (pas de joblib ici pour simplifier) ---
        i, j = np.random.choice(len(pop_tours), 2, replace=False)
        ctour = solver.ox_crossover(pop_tours[i], pop_tours[j])
        croutes = solver.split_vrptw(inst, ctour) if inst.has_tw else solver.split_cvrp(inst, ctour)
        if croutes is None:
            q = max(2, inst.n//20)
            removed = solver.shaw_removal(inst, best, q) if np.random.rand()<0.7 else solver.random_removal(inst, best, q)
            cand = solver.remove_customers(best, removed)
            cand = solver.regret_repair(inst, cand, removed)
            ctour = solver.tour_from_routes(cand)
            croutes = (solver.split_vrptw(inst, ctour) if inst.has_tw else solver.split_cvrp(inst, ctour)) or cand
        croutes = solver.rvnd(inst, croutes, max_loops=3, nn=nn)
        cc = fit(croutes)
        if cc < bestc - 1e-9:
            best, bestc = croutes, cc
        best_cost_log.append(bestc)  # 🔹 log du meilleur coût trouvé à cette itération
    return best

# On remplace temporairement la fonction originale
solver.hgs_solve = patched_hgs_solve

# --- Exécution ---
print(f"[RUN] Lancement convergence sur {INSTANCE_PATH.name} ...")
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1"
})
solver.solve_file(str(INSTANCE_PATH), loops=LOOPS, nnk=NNK, workers=1, fast=True, pop=56)

# --- Export CSV ---
df = pd.DataFrame({"iter": range(len(best_cost_log)), "best_cost": best_cost_log})
df.to_csv(OUTPUT_CSV, index=False)
print(f"[OK] Données enregistrées dans {OUTPUT_CSV}")

# --- Tracé de la courbe ---
plt.figure(figsize=(7,4))
plt.plot(df["iter"], df["best_cost"], color="royalblue", linewidth=2)
plt.xlabel("Itérations (loops)")
plt.ylabel("Meilleur coût trouvé")
plt.title(f"Courbe de convergence - {INSTANCE_PATH.name}")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"[OK] Figure enregistrée → {OUTPUT_PNG}")
plt.show()

# --- Restaure la fonction originale ---
solver.hgs_solve = original_hgs_solve