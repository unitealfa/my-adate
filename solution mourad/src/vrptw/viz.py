from __future__ import annotations
import matplotlib.pyplot as plt
from .data import ProblemData
from .solution import Solution

def plot_routes(data: ProblemData, sol: Solution, path_png: str):
    xs = [data.depot.x] + [c.x for c in data.clients]
    ys = [data.depot.y] + [c.y for c in data.clients]
    plt.figure(figsize=(10,7))
    plt.scatter([xs[0]],[ys[0]], marker='s', s=120)
    for r in sol.routes:
        rx = [xs[i] for i in r]
        ry = [ys[i] for i in r]
        plt.plot(rx, ry, marker='o', linewidth=1)
    plt.title("Routes VRPTW")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
