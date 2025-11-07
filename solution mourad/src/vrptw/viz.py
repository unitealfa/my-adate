from __future__ import annotations
import matplotlib.pyplot as plt
from .data import ProblemData
from .solution import Solution


def _route_length(route: list[int], dist_matrix) -> float:
    """Returns the total travel distance of a route."""

    if not route:
        return 0.0

    full_route = list(route)
    if full_route[0] != 0:
        full_route.insert(0, 0)
    if full_route[-1] != 0:
        full_route.append(0)

    length = 0.0
    prev = full_route[0]
    for nxt in full_route[1:]:
        length += dist_matrix[prev, nxt]
        prev = nxt

    return length


def _compute_figsize(n_clients: int) -> tuple[float, float]:
    """Computes a figsize that scales with the instance size."""

    base = 6.0
    scale = min(16.0, base + 0.18 * max(0, n_clients))
    return (scale, scale * 0.7)


def plot_routes(data: ProblemData, sol: Solution, path_png: str, show: bool = False):
    xs = [data.depot.x] + [c.x for c in data.clients]
    ys = [data.depot.y] + [c.y for c in data.clients]

    fig, ax = plt.subplots(figsize=_compute_figsize(data.n_clients))
    ax.scatter([xs[0]], [ys[0]], marker="s", s=120, label="Dépôt")

    longest_route_idx = max(
        range(len(sol.routes)),
        key=lambda idx: _route_length(sol.routes[idx], data.dist),
        default=None,
    )

    longest_color = None
    for idx, route in enumerate(sol.routes):
        rx = [xs[i] for i in route]
        ry = [ys[i] for i in route]
        linewidth = 3 if idx == longest_route_idx else 1.2
        zorder = 3 if idx == longest_route_idx else 2
        (line,) = ax.plot(rx, ry, marker="o", linewidth=linewidth, zorder=zorder)
        if idx == longest_route_idx:
            longest_color = line.get_color()

    if longest_route_idx is not None and longest_color is not None:
        ax.plot([], [], linewidth=3, color=longest_color, label="Tournée la plus longue")

    ax.set_title("Routes VRPTW")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
