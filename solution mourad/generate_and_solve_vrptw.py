"""Utility script to generate a random VRPTW instance and solve it with PyVRP.

The script interactively asks for the number of vehicles and clients, randomly
samples a VRPTW instance, stores the generated data on disk, and immediately
runs PyVRP's solver on the generated instance. Optional visualisations of the
instance and solution are stored alongside the generated input when matplotlib
is available.

Usage
-----
$ python generate_and_solve_vrptw.py
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ensure the local pyvrp package is importable when running from the source tree.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyvrp import read  # type: ignore  # noqa: E402
from pyvrp.solve import solve  # type: ignore  # noqa: E402
from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria, NoImprovement  # type: ignore  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "generated_instances"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ClientData:
    """Simple container for generated client information."""

    x: int
    y: int
    demand: int
    service: int
    ready: int
    due: int


@dataclass
class InstanceData:
    """Container for the generated VRPTW instance attributes."""

    name: str
    num_vehicles: int
    vehicle_capacity: int
    depot: ClientData
    clients: list[ClientData]
    seed: int

    @property
    def dimension(self) -> int:
        return len(self.clients) + 1  # depot + clients


def prompt_positive_int(label: str) -> int:
    """Prompts the user until a strictly positive integer is provided."""

    while True:
        raw = input(f"Veuillez saisir {label} : ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Entrée invalide, merci de saisir un entier.")
            continue

        if value <= 0:
            print("La valeur doit être strictement positive.")
            continue

        return value


def generate_clients(
    rng: random.Random, num_clients: int, depot: ClientData, horizon: int
) -> list[ClientData]:
    """Generates random clients within a square bounding box."""

    clients: list[ClientData] = []
    depot_coords = (depot.x, depot.y)

    for _ in range(num_clients):
        x = rng.randint(0, 100)
        y = rng.randint(0, 100)
        demand = rng.randint(1, 10)
        service = rng.randint(5, 20)

        earliest = rng.randint(0, horizon // 2)
        slack = rng.randint(60, 180)
        travel_buffer = int(math.hypot(x - depot_coords[0], y - depot_coords[1]))
        due = min(horizon, earliest + slack + travel_buffer)

        # Ensure feasibility with respect to service time.
        due = max(due, earliest + service)

        clients.append(ClientData(x, y, demand, service, earliest, due))

    return clients


def compute_vehicle_capacity(num_vehicles: int, clients: Iterable[ClientData]) -> int:
    """Chooses a capacity that is likely to make the instance solvable."""

    total_demand = sum(client.demand for client in clients)
    avg_per_vehicle = math.ceil(total_demand / num_vehicles)
    capacity = max(30, math.ceil(avg_per_vehicle * 1.2))
    return capacity


def build_instance(num_vehicles: int, num_clients: int, seed: int) -> InstanceData:
    """Creates an :class:`InstanceData` with randomised attributes."""

    rng = random.Random(seed)
    depot = ClientData(x=50, y=50, demand=0, service=0, ready=0, due=600)
    clients = generate_clients(rng, num_clients, depot, horizon=600)
    capacity = compute_vehicle_capacity(num_vehicles, clients)

    name = f"VRPTW_{num_vehicles}veh_{num_clients}cust"
    return InstanceData(
        name=name,
        num_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        depot=depot,
        clients=clients,
        seed=seed,
    )


def write_vrplib(path: Path, instance: InstanceData) -> None:
    """Writes the generated instance to disk using the VRPLIB format."""

    lines: list[str] = [
        f"NAME: {instance.name}",
        "TYPE: VRPTW",
        f"DIMENSION: {instance.dimension}",
        f"VEHICLES: {instance.num_vehicles}",
        f"CAPACITY: {instance.vehicle_capacity}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]

    # Depot coordinates (index 1)
    lines.append(f"1 {instance.depot.x} {instance.depot.y}")

    # Client coordinates start at index 2
    for idx, client in enumerate(instance.clients, start=2):
        lines.append(f"{idx} {client.x} {client.y}")

    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    for idx, client in enumerate(instance.clients, start=2):
        lines.append(f"{idx} {client.demand}")

    lines.append("SERVICE_TIME_SECTION")
    lines.append("1 0")
    for idx, client in enumerate(instance.clients, start=2):
        lines.append(f"{idx} {client.service}")

    lines.append("TIME_WINDOW_SECTION")
    lines.append(f"1 {instance.depot.ready} {instance.depot.due}")
    for idx, client in enumerate(instance.clients, start=2):
        lines.append(f"{idx} {client.ready} {client.due}")

    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(path: Path, instance: InstanceData) -> None:
    """Stores additional metadata (JSON) for the generated instance."""

    import json

    metadata = {
        "name": instance.name,
        "num_vehicles": instance.num_vehicles,
        "num_clients": len(instance.clients),
        "vehicle_capacity": instance.vehicle_capacity,
        "seed": instance.seed,
        "clients": [
            {
                "x": client.x,
                "y": client.y,
                "demand": client.demand,
                "service": client.service,
                "time_window": [client.ready, client.due],
            }
            for client in instance.clients
        ],
    }

    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def maybe_plot_outputs(base_path: Path, data, result) -> None:
    """Generates optional visualisations if matplotlib is available."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from pyvrp.plotting import plot_instance, plot_result  # type: ignore
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Visualisation non disponible ({exc}).")
        return

    fig = plt.figure(figsize=(8, 6))
    plot_instance(data, fig=fig)
    instance_fig = base_path.with_suffix(".instance.png")
    fig.savefig(instance_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Carte de l'instance sauvegardée dans {instance_fig}.")

    fig = plt.figure(figsize=(8, 6))
    plot_result(result, data, fig=fig)
    solution_fig = base_path.with_suffix(".solution.png")
    fig.savefig(solution_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Solution visualisée dans {solution_fig}.")


def run_solver(instance_path: Path) -> None:
    """Reads the generated instance and solves it using PyVRP."""

    data = read.read(instance_path)

    stopping = MultipleCriteria(
        [
            MaxRuntime(10),
            MaxIterations(2_000),
            NoImprovement(max_iterations=200),
        ]
    )

    seed = random.randint(0, 1_000_000)
    result = solve(data, stop=stopping, seed=seed, collect_stats=True, display=False)

    print("\n" + result.summary())

    maybe_plot_outputs(instance_path.with_suffix(""), data, result)

    summary_path = instance_path.with_suffix(".summary.txt")
    summary_path.write_text(str(result) + "\n", encoding="utf-8")
    print(f"Résumé détaillé sauvegardé dans {summary_path}.")


def main() -> None:
    print("=== Générateur et solveur VRPTW (PyVRP) ===")
    num_vehicles = prompt_positive_int("le nombre de camions")
    num_clients = prompt_positive_int("le nombre de clients")

    seed = random.randint(0, 1_000_000)
    instance = build_instance(num_vehicles, num_clients, seed)

    base_name = f"{instance.name}"
    instance_path = OUTPUT_DIR / f"{base_name}.vrplib"
    metadata_path = OUTPUT_DIR / f"{base_name}.json"

    write_vrplib(instance_path, instance)
    write_metadata(metadata_path, instance)

    print(
        f"Instance VRPTW générée : {instance_path}\n"
        f"Métadonnées associées : {metadata_path}\n"
        "Lancement du solveur PyVRP..."
    )

    run_solver(instance_path)


if __name__ == "__main__":
    main()