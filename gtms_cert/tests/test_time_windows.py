from collections import defaultdict

import pytest

from gtms_cert.geo import DistanceOracle


def _build_oracle_with_windows() -> DistanceOracle:
    matrix = [
        [0.0, 30.0, 60.0],
        [30.0, 0.0, 40.0],
        [60.0, 40.0, 0.0],
    ]
    return DistanceOracle(
        node_ids=["D0", "C1", "C2"],
        depot_id="D0",
        metric="matrix",
        avg_speed_kmh=None,
        coordinates=None,
        matrix=matrix,
        symmetric=True,
        candidate_edges=defaultdict(list),
        time_windows={"C1": (50.0, 120.0), "C2": (200.0, 260.0)},
    )


def test_route_time_includes_waiting_for_time_windows():
    oracle = _build_oracle_with_windows()
    route = ["D0", "C1", "C2", "D0"]
    # Travel details:
    # D0 -> C1: 30 minutes, wait until 50 ( +20 )
    # C1 -> C2: 40 minutes, arrive at 90, wait until 200 ( +110 )
    # C2 -> D0: 60 minutes
    expected = 260.0
    assert oracle.route_time(route) == pytest.approx(expected)


def test_route_time_penalises_late_arrivals():
    matrix = [
        [0.0, 50.0],
        [50.0, 0.0],
    ]
    oracle = DistanceOracle(
        node_ids=["D0", "C1"],
        depot_id="D0",
        metric="matrix",
        avg_speed_kmh=None,
        coordinates=None,
        matrix=matrix,
        symmetric=True,
        candidate_edges=defaultdict(list),
        time_windows={"C1": (0.0, 30.0)},
        lateness_penalty_multiplier=2.0,
    )
    route = ["D0", "C1", "D0"]
    base_travel = oracle.get("D0", "C1") + oracle.get("C1", "D0")
    lateness = 20.0
    expected = base_travel + lateness * oracle.lateness_penalty_multiplier
    assert oracle.route_time(route) == pytest.approx(expected)
