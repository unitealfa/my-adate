from __future__ import annotations
import matplotlib.pyplot as plt
from .data import ProblemData
from .solution import Solution


def plot_routes(data: ProblemData, sol: Solution, path_png: str, show: bool = False):
    xs = [data.depot.x] + [c.x for c in data.clients]
    ys = [data.depot.y] + [c.y for c in data.clients]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter([xs[0]], [ys[0]], marker="s", s=120)
    for r in sol.routes:
        rx = [xs[i] for i in r]
        ry = [ys[i] for i in r]
        ax.plot(rx, ry, marker="o", linewidth=1)
    ax.set_title("Routes VRPTW")
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
