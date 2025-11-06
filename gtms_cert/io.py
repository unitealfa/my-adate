"""Input/output helpers for the GTMS-Cert solver."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .geo import DistanceOracle, build_distance_oracle


@dataclass
class ProblemData:
    """Structured view of an mTSP instance."""

    depot_id: str
    vehicle_count: int
    use_all: bool
    clients: List[str]
    oracle: DistanceOracle
    time_windows: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def read_input(path: str | Path, cands: int = 32) -> ProblemData:
    """Read and validate the instance description."""
    payload = json.loads(Path(path).read_text())
    if "vehicles" not in payload or "k" not in payload["vehicles"]:
        raise ValueError("Missing vehicle information in input JSON")
    if "use_all" not in payload["vehicles"]:
        raise ValueError("vehicles.use_all flag is required")

    if "depot" in payload:
        depot = payload["depot"].get("id")
        if depot is None:
            raise ValueError("Depot entry must contain an 'id'")
        clients = [c["id"] for c in payload.get("customers", [])]
    else:
        depot = payload.get("depot_id")
        if depot is None:
            raise ValueError("Missing depot identifier")
        clients = [i for i in payload.get("node_ids", []) if i != depot]

    k = int(payload["vehicles"]["k"])
    use_all = bool(payload["vehicles"].get("use_all", False))
    if use_all and len(clients) < k:
        raise ValueError("Number of clients must be >= number of vehicles when use_all is true")

    oracle = build_distance_oracle(payload, cands=cands)
    time_windows = dict(getattr(oracle, "time_windows", {}))
    return ProblemData(
        depot_id=depot,
        vehicle_count=k,
        use_all=use_all,
        clients=clients,
        oracle=oracle,
        time_windows=time_windows,
    )


def build_output_payload(
    routes: Sequence[Sequence[str]],
    oracle: DistanceOracle,
    ub: float,
    lb: float,
    gap: float,
    longest_route_idx: int,
) -> Dict[str, object]:
    """Create a serialisable dictionary describing the solution."""
    details = []
    for vehicle, seq in enumerate(routes, start=1):
        time_min = oracle.route_time(seq)
        details.append({"vehicle": vehicle, "sequence": list(seq), "time_min": time_min})
    payload: Dict[str, object] = {
        "makespan": ub,
        "best_upper_bound": ub,
        "best_lower_bound": lb,
        "gap": gap,
        "routes": details,
        "longest_route_vehicle": longest_route_idx + 1,
    }
    return payload


def write_output(
    path: str | Path,
    routes: Sequence[Sequence[str]],
    oracle: DistanceOracle,
    ub: float,
    lb: float,
    gap: float,
    longest_route_idx: int,
) -> Dict[str, object]:
    """Serialise the solution to JSON and return the payload."""
    payload = build_output_payload(routes, oracle, ub, lb, gap, longest_route_idx)
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def generate_random_payload(
    vehicle_count: int,
    client_count: int,
    seed: int,
    *,
    include_time_windows: bool = True,
) -> Dict[str, object]:
    """Create a random instance payload matching the solver's expected format.

    Each generated customer receives a realistic delivery time window in minutes
    from the start of the planning horizon when ``include_time_windows`` is
    ``True``.
    """
    if vehicle_count <= 0:
        raise ValueError("vehicle_count must be strictly positive")
    if client_count < vehicle_count:
        raise ValueError("client_count must be >= vehicle_count")

    rng = random.Random(seed)
    depot_id = "D0"
    base_lat, base_lon = 48.8566, 2.3522  # Around Paris to keep coordinates realistic
    spread = 0.25  # Approx ~25km in each direction

    def _random_coord() -> tuple[float, float]:
        return (
            base_lat + rng.uniform(-spread, spread),
            base_lon + rng.uniform(-spread, spread),
        )

    depot_coord = _random_coord()
    day_start = 8 * 60  # 08:00 in minutes
    day_end = 19 * 60  # 19:00 in minutes
    min_window = 90
    max_window = 180

    customers = []
    for i in range(client_count):
        window_start = rng.uniform(day_start, max(day_start, day_end - min_window))
        window_length = rng.uniform(min_window, max_window)
        window_end = min(window_start + window_length, day_end)
        customer_payload = {
            "id": f"C{i+1}",
            "coord": _random_coord(),
        }
        if include_time_windows:
            customer_payload["time_window"] = [
                round(window_start, 2),
                round(window_end, 2),
            ]
        customers.append(customer_payload)

    return {
        "depot": {"id": depot_id, "coord": depot_coord},
        "customers": customers,
        "vehicles": {"k": vehicle_count, "use_all": True},
        "metric": {"type": "haversine", "avg_speed_kmh": 40.0},
        "symmetric": True,
    }


def generate_random_problem(
    vehicle_count: int,
    client_count: int,
    seed: int,
    *,
    cands: int = 32,
    include_time_windows: bool = True,
) -> ProblemData:
    """Generate a random mTSP instance compatible with the solver pipeline."""
    payload = generate_random_payload(
        vehicle_count,
        client_count,
        seed,
        include_time_windows=include_time_windows,
    )
    oracle = build_distance_oracle(payload, cands=cands)
    clients = [customer["id"] for customer in payload["customers"]]
    time_windows = dict(getattr(oracle, "time_windows", {}))
    depot_id = payload["depot"]["id"]
    return ProblemData(
        depot_id=depot_id,
        vehicle_count=vehicle_count,
        use_all=True,
        clients=clients,
        oracle=oracle,
        time_windows=time_windows,
    )
