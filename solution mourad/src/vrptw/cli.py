from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import random
from .data import build_data, Depot, Client, save_dataset
from .solver import solve
from .viz import plot_routes

def parse_args():
    p = argparse.ArgumentParser(description="VRPTW (k camions, TW dures, durée max)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data", type=str, help="JSON existant (clients + TW)")
    mode.add_argument("--random", action="store_true", help="générer aléatoirement")

    p.add_argument("--n-clients", type=int, default=50)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--shift-duration", type=float, default=None)
    p.add_argument("--time-limit", type=int, default=60)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def generate_random(n_clients: int, seed: int, out_dir: Path) -> str:
    random.seed(seed)
    depot = Depot(0.0, 0.0, 0, 10**9, 0)
    clients = []
    for i in range(1, n_clients+1):
        x, y = random.uniform(-10,10), random.uniform(-10,10)
        width = random.randint(60, 180)
        e = random.randint(0, 600-width)
        l = e + width
        clients.append(Client(i, x, y, e, l, 5))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"vrptw_{ts}_{n_clients}c.json"
    save_dataset(depot, clients, str(path))
    return str(path)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.data:
        path_json = args.data
    else:
        path_json = generate_random(args.n_clients, args.seed, Path("data/generated"))

    data, best = solve(path_json, k=args.k, shift_duration=args.shift_duration, time_limit_s=args.time_limit)

    # sorties
    print(f"Objective: {best.cost:.3f} | Dist: {best.dist:.3f} | TW_violation: {best.time_warp:.3f} | LastReturn: {best.last_return:.3f}")
    png = out_dir / "routes.png"
    plot_routes(data, best, str(png))
    print(f"Plot écrit: {png}")

if __name__ == "__main__":
    main()
