
import os, re, time, itertools, sys, importlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============== USER CONFIG ===============
# Change this if your solver file is named differently (without .py):
SOLVER_MODULE = "vrp_solver"

# Paths to instances (adjust if your dataset is elsewhere)
C101 = Path("data/cvrplib/Vrp-Set-Solomon/C101.txt")
R102 = Path("data/cvrplib/Vrp-Set-Solomon/R102.txt")

# DOE grid (keep it small and meaningful for static study)
LOOPS = [300, 500, 700]
NNK   = [20, 25]  # k-nearest-neighbors used by 'fast' mode in your solver

# Fixed parameters (static baseline spirit)
FIXED = dict(
    pop=None,            # None -> auto_params decides; or set e.g., 56
    seed=12345,          # fixed seed for reproducibility
    fast=True,           # build granular neighborhoods
    init='auto',         # your solver: 'regret' for TW, 'sweep' for CVRP
    workers=1,           # single worker to avoid nondeterminism
    time_limit=None      # or fixed (e.g., 60) if desired
)
# ==========================================

# Lock math libs to 1 thread (reproducibility)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

def _import_solve(module_name: str):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[ERROR] Cannot import module '{module_name}': {e}", file=sys.stderr)
        sys.exit(1)
    if not hasattr(mod, "solve_file"):
        print(f"[ERROR] Module '{module_name}' has no function solve_file(path, ...).", file=sys.stderr)
        sys.exit(1)
    return mod.solve_file

def parse_solver_text(txt: str) -> Dict[str, Any]:
    """
    Expected lines include:
      - 'Cost 1234'
      - 'Time 12.34s'
      - 'Temps du plus long trajet 567.89s' (optional)
      - 'gap 1.23% (ref 1000)' or 'gap N/A'
    """
    # Cost
    m_cost = re.search(r'(?i)\bcost\b\s+([0-9]+(?:\.[0-9]+)?)', txt)
    cost = float(m_cost.group(1)) if m_cost else np.nan

    # Total time
    m_time = re.search(r'(?i)\btime\b\s+([0-9]+(?:\.[0-9]+)?)s', txt)
    tsec = float(m_time.group(1)) if m_time else np.nan

    # Longest route time (optional)
    m_lrt = re.search(r'(?i)plus long trajet\s+([0-9]+(?:\.[0-9]+)?)s', txt)
    lrt = float(m_lrt.group(1)) if m_lrt else np.nan

    # GAP
    m_gap = re.search(r'(?i)\bgap\b\s+([0-9]+(?:\.[0-9]+)?)\s*%', txt)
    gap = float(m_gap.group(1)) if m_gap else np.nan

    # Ref (optional)
    m_ref = re.search(r'\(ref\s+([0-9]+(?:\.[0-9]+)?)\)', txt, flags=re.I)
    ref = float(m_ref.group(1)) if m_ref else np.nan

    return dict(total_cost=cost, time_s=tsec, longest_route_time_s=lrt, gap_pct=gap, ref_cost=ref)

def run_one(solve_file, path: Path, loops: int, nnk: int, fixed: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Instance not found: {path}")

    # Build kwargs for your solver
    kwargs = dict(
        loops=loops,
        pop=fixed["pop"],
        seed=fixed["seed"],
        fast=fixed["fast"],
        nnk=nnk,
        init=fixed["init"],
        workers=fixed["workers"],
        time_limit=fixed["time_limit"],
    )
    # Remove None keys (solver handles auto_params)
    kwargs = {k:v for k,v in kwargs.items() if v is not None}

    t0 = time.time()
    out = solve_file(str(path), **kwargs)
    t1 = time.time()

    parsed = parse_solver_text(out)
    rec = dict(
        instance=path.name,
        loops=loops,
        nnk=nnk,
        seed=fixed["seed"],
        workers=fixed["workers"],
        fast=fixed["fast"],
        wallclock_s=round(t1-t0, 3),
    )
    rec.update(parsed)
    return rec

def main():
    solve_file = _import_solve(SOLVER_MODULE)

    instances = [C101, R102]
    grid = list(itertools.product(LOOPS, NNK))

    rows: List[Dict[str, Any]] = []
    for inst in instances:
        for (L, K) in grid:
            try:
                rec = run_one(solve_file, inst, loops=L, nnk=K, fixed=FIXED)
            except Exception as e:
                rec = dict(instance=inst.name, loops=L, nnk=K, error=str(e))
            rows.append(rec)

    df = pd.DataFrame(rows)
    csv_path = Path("static_workshop_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV → {csv_path.resolve()}")

    # ---- Plots ----
    # (1) Bar plot GAP by instance/config
    try:
        df2 = df.dropna(subset=["gap_pct"]).copy()
        df2["label"] = df2.apply(lambda r: f"L{int(r['loops'])}/k{int(r['nnk'])}", axis=1)
        df2["cat"] = df2.apply(lambda r: f"{r['instance']} ({int(r['loops'])}/{int(r['nnk'])})", axis=1)

        plt.figure()
        plt.bar(df2["cat"], df2["gap_pct"])
        plt.title("GAP% by Instance and (loops/nnk)")
        plt.xlabel("Instance (loops/nnk)")
        plt.ylabel("GAP (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("gap_bar.png", dpi=150)
        plt.close()
        print("[OK] Saved figure → gap_bar.png")
    except Exception as e:
        print(f"[WARN] Could not create gap_bar.png: {e}")

    # (2) Scatter Time vs GAP (amélioré)
    try:
        df3 = df.dropna(subset=["gap_pct","time_s"]).copy()

        # Palette de couleurs distinctes pour chaque instance
        instances_unique = df3["instance"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(instances_unique)))  # tab10 = 10 couleurs lisibles
        color_map = dict(zip(instances_unique, colors))

        plt.figure(figsize=(7,5))

        # Tracer chaque instance avec une couleur différente
        for inst_name in instances_unique:
            sub = df3[df3["instance"] == inst_name]
            plt.scatter(sub["time_s"], sub["gap_pct"],
                        label=inst_name,
                        color=color_map[inst_name],
                        s=70, edgecolor="black", alpha=0.8)
            # Ajouter les étiquettes (configurables)
            for _, r in sub.iterrows():
                lbl = f"L{int(r['loops'])}/k{int(r['nnk'])}"
                plt.annotate(lbl, (r["time_s"], r["gap_pct"]),
                             xytext=(4,4), textcoords="offset points",
                             fontsize=8, color="black")

        plt.title("Time (s) vs GAP (%) by Instance")
        plt.xlabel("Solver-reported time (s)")
        plt.ylabel("GAP (%)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Instance", loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig("time_vs_gap.png", dpi=150)
        plt.close()
        print("[OK] Saved improved figure → time_vs_gap.png")
    except Exception as e:
        print(f"[WARN] Could not create improved time_vs_gap.png: {e}")


if __name__ == "__main__":
    main()
