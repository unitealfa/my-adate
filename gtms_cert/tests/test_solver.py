import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gtms_cert.main import solve_gtms_cert


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


def test_small_matrix_gap_under_one_percent(tmp_path):
    instance = _build_constant_matrix_instance(n_clients=3, k=3, weight=10.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(instance))
    solve_gtms_cert(str(input_path), str(output_path), seed=0, lb_iters=100)
    result = json.loads(output_path.read_text())
    assert len(result["routes"]) == 3
    assert all(len(route["sequence"]) == 3 for route in result["routes"])
    assert result["gap"] <= 0.01 + 1e-6


def test_medium_matrix_balanced_partition(tmp_path):
    instance = _build_constant_matrix_instance(n_clients=200, k=10, weight=7.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(instance))
    solve_gtms_cert(str(input_path), str(output_path), seed=0, lb_iters=150)
    result = json.loads(output_path.read_text())
    assert len(result["routes"]) == 10
    assert all(route["sequence"][0] == "D0" and route["sequence"][-1] == "D0" for route in result["routes"])
    assert result["gap"] <= 0.01 + 1e-6


def test_large_matrix_scaling(tmp_path):
    instance = _build_constant_matrix_instance(n_clients=512, k=64, weight=5.0)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(instance))
    solve_gtms_cert(str(input_path), str(output_path), seed=0, lb_iters=120)
    result = json.loads(output_path.read_text())
    assert len(result["routes"]) == 64
    assert result["gap"] <= 0.01 + 1e-6
