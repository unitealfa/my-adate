import os
from solver.data import load_vrplib
from solver.solver import solve_vrp

def main():
    # Chemin vers le dossier data (relatif à ce fichier)
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    path_vrp = os.path.join(data_dir, "X-n101-k25.vrp")

    inst = load_vrplib(path_vrp)
    res = solve_vrp(inst, rng_seed=11, tabu_max_iter=2000, tabu_no_improve=250)

    print("Instance:", inst.name)
    print("Feasible:", res["feasible"], "| Vehicles:", res["used_vehicles"], "| Cost:", f"{res['cost']:.2f}")
    # Afficher les 3 premières routes
    for idx, r in enumerate(res["routes"][:3]):
        print(f"Route {idx+1} ({len(r)} clients):", r)

if __name__ == "__main__":
    main()
