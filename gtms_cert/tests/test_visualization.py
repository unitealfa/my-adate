import json
import tempfile
from pathlib import Path

from gtms_cert.io import read_input
from gtms_cert.visualize import build_route_timelines, compute_node_positions


def _build_constant_matrix_instance(n_clients: int, k: int, weight: float) -> dict:
    node_ids = ["D0"] + [f"C{i}" for i in range(1, n_clients + 1)]
    size = len(node_ids)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(1, size):
        matrix[0][i] = matrix[i][0] = weight
    for i in range(1, size):
        for j in range(1, size):
            if i == j:
                continue
            matrix[i][j] = matrix[0][i] + matrix[0][j]
    return {
        "node_ids": node_ids,
        "depot_id": "D0",
        "vehicles": {"k": k, "use_all": True},
        "time_matrix_min": matrix,
        "symmetric": True,
        "objective": "minimize_makespan",
    }


def _write_payload(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix="_visu_test.json", delete=False, encoding="utf-8")
    json.dump(payload, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def test_compute_node_positions_unique_coordinates():
    payload = _build_constant_matrix_instance(6, 3, 7.0)
    temp_path = _write_payload(payload)
    try:
        data = read_input(temp_path, cands=4)
    finally:
        temp_path.unlink(missing_ok=True)

    positions = compute_node_positions(data.oracle)
    assert len(positions) == len(data.oracle.node_ids)
    # Ensure that each node has a distinct position to avoid overlapping trucks.
    assert len({tuple(map(float, coord)) for coord in positions.values()}) == len(positions)
    assert data.depot_id in positions


def test_build_route_timelines_matches_oracle_times():
    payload = _build_constant_matrix_instance(4, 2, 5.0)
    temp_path = _write_payload(payload)
    try:
        data = read_input(temp_path, cands=3)
    finally:
        temp_path.unlink(missing_ok=True)

    route = [data.depot_id, "C1", "C2", data.depot_id]
    timelines = build_route_timelines([route], data.oracle)
    assert len(timelines) == 1
    timeline = timelines[0]
    assert timeline.route == route
    assert timeline.total_time == data.oracle.route_time(route)
    assert all(segment.travel_time > 0 for segment in timeline.segments)
