"""Input/output helpers for the GTMS-Cert solver."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .geo import DistanceOracle, build_distance_oracle


@dataclass
class ProblemData:
    """Structured view of an mTSP instance."""

    depot_id: str
    vehicle_count: int
    use_all: bool
    clients: List[str]
    oracle: DistanceOracle


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
    return ProblemData(depot_id=depot, vehicle_count=k, use_all=use_all, clients=clients, oracle=oracle)


def write_output(
    path: str | Path,
    routes: Sequence[Sequence[str]],
    oracle: DistanceOracle,
    ub: float,
    lb: float,
    gap: float,
    longest_route_idx: int,
) -> None:
    """Serialise the solution to JSON."""
    details = []
    for vehicle, seq in enumerate(routes, start=1):
        time_min = oracle.route_time(seq)
        details.append({"vehicle": vehicle, "sequence": list(seq), "time_min": time_min})
    payload = {
        "makespan": ub,
        "best_upper_bound": ub,
        "best_lower_bound": lb,
        "gap": gap,
        "routes": details,
        "longest_route_vehicle": longest_route_idx + 1,
    }
    Path(path).write_text(json.dumps(payload, indent=2))
