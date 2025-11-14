#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from k import solve_file

# === Instance à tester ===
instance_path = "solution mourad/data/cvrplib/Vrp-Set-Solomon/C101.txt"

print("[INFO] Exécution du solveur...")
best_cost, best_gap, convergence = solve_file(instance_path, return_raw=True)

print(f"[INFO] Cost final : {best_cost}")
print(f"[INFO] GAP final : {best_gap:.2f}%")
print(f"[INFO] Points collectés : {len(convergence)}")

# === Sauvegarde CSV ===
df = pd.DataFrame({
    "iteration": list(range(len(convergence))),
    "cost": convergence
})

df.to_csv("convergence_C101.csv", index=False)
print("[INFO] Données sauvegardées dans convergence_C101.csv")

# === Tracer la courbe ===
plt.figure(figsize=(8,5))
plt.plot(df["iteration"], df["cost"], linewidth=2)
plt.title("Courbe de convergence - C101")
plt.xlabel("Itérations")
plt.ylabel("Coût")
plt.grid(True)
plt.tight_layout()
plt.show()
