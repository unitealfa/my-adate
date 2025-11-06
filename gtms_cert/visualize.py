"""Interactive visualisation of GTMS-Cert test scenarios."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .io import ProblemData, build_output_payload, read_input
from .main import solve_gtms_cert, solve_problem_data


Coordinate = Tuple[float, float]


@dataclass
class RouteSegment:
    """A single segment travelled by a truck."""

    start: str
    end: str
    travel_time: float


@dataclass
class RouteTimeline:
    """Timeline describing how a truck moves through its route."""

    route: List[str]
    segments: List[RouteSegment]
    total_time: float


@dataclass
class TestResult:
    """Summary of a regression test execution."""

    name: str
    passed: bool
    message: str


def compute_node_positions(oracle) -> Dict[str, Coordinate]:
    """Return 2D positions for each node in the instance."""

    nodes = oracle.node_ids
    if not nodes:
        return {}

    if oracle.coordinates:
        coords = {node: tuple(oracle.coordinates[node]) for node in nodes}
        # Normalise coordinates to a centred layout.
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        return {node: (coords[node][0] - cx, coords[node][1] - cy) for node in nodes}

    radius = max(5.0, len(nodes) / math.pi)
    positions: Dict[str, Coordinate] = {}
    for idx, node in enumerate(nodes):
        angle = (2 * math.pi * idx) / len(nodes)
        positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def build_route_timelines(
    routes: Sequence[Sequence[str]], oracle
) -> List[RouteTimeline]:
    """Pre-compute travel times for each route."""

    timelines: List[RouteTimeline] = []
    for route in routes:
        seq = list(route)
        if seq and seq[0] != seq[-1]:
            seq.append(seq[0])
        segments: List[RouteSegment] = []
        total = 0.0
        for start, end in zip(seq[:-1], seq[1:]):
            travel = float(oracle.get(start, end))
            segments.append(RouteSegment(start=start, end=end, travel_time=travel))
            total += travel
        timelines.append(RouteTimeline(route=seq, segments=segments, total_time=total))
    return timelines


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


def _write_temp_instance(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix="_gtms_visu.json", delete=False, encoding="utf-8"
    )
    json.dump(payload, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _run_scenario(
    name: str,
    payload: dict,
    *,
    seed: int,
    cands: int,
    lb_iters: int,
    expected_routes: int,
    expect_gap: float,
) -> tuple[TestResult, ProblemData, dict]:
    safe_payload = json.loads(json.dumps(payload))
    temp_path = _write_temp_instance(safe_payload)
    output_tmp = tempfile.NamedTemporaryFile(
        "w", suffix="_gtms_result.json", delete=False, encoding="utf-8"
    )
    output_tmp.close()
    output_path = Path(output_tmp.name)
    data = read_input(temp_path, cands=cands)
    try:
        result = solve_gtms_cert(
            str(temp_path),
            str(output_path),
            seed=seed,
            cands=cands,
            lb_iters=lb_iters,
        )
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass

    passed = True
    message_parts: List[str] = []

    if len(result.get("routes", [])) != expected_routes:
        passed = False
        message_parts.append(
            f"Routes attendues: {expected_routes}, obtenues: {len(result.get('routes', []))}"
        )

    gap = float(result.get("gap", 0.0))
    if gap > expect_gap + 1e-6:
        passed = False
        message_parts.append(f"Gap {gap:.4f} > {expect_gap:.4f}")

    if not message_parts:
        message_parts.append("Succès")

    return TestResult(name=name, passed=passed, message="; ".join(message_parts)), data, result


def run_regression_tests(cands: int) -> tuple[List[TestResult], ProblemData, dict]:
    """Execute the reference tests and return their summaries.

    The medium scenario (200 clients) is returned for visualisation.
    """

    tests: List[TestResult] = []
    medium_data: ProblemData | None = None
    medium_result: dict | None = None

    medium_payload_path = Path(__file__).resolve().parent / "tests" / "sample_instance_200_clients.json"
    medium_payload = json.loads(medium_payload_path.read_text(encoding="utf-8"))
    medium_payload["vehicles"]["k"] = 10

    scenario_payloads = [
        (
            "Petit scénario (3 clients)",
            _build_constant_matrix_instance(3, 3, 10.0),
            0,
            100,
            3,
            0.01,
            False,
        ),
        (
            "Scénario moyen (200 clients)",
            medium_payload,
            0,
            150,
            10,
            0.01,
            True,
        ),
        (
            "Grand scénario (512 clients)",
            _build_constant_matrix_instance(512, 64, 5.0),
            0,
            120,
            64,
            0.01,
            False,
        ),
    ]

    for name, payload, seed, lb_iters, expected_routes, expect_gap, is_medium in scenario_payloads:
        test_result, data, result = _run_scenario(
            name,
            payload,
            seed=seed,
            cands=cands,
            lb_iters=lb_iters,
            expected_routes=expected_routes,
            expect_gap=expect_gap,
        )
        tests.append(test_result)
        if is_medium:
            medium_data = data
            medium_result = result

    if medium_data is None or medium_result is None:
        raise RuntimeError("Scénario moyen introuvable")

    return tests, medium_data, medium_result


def _format_test_summary(tests: Sequence[TestResult]) -> str:
    if not tests:
        return ""
    lines = ["Résultats des tests :"]
    for test in tests:
        status = "✅" if test.passed else "❌"
        lines.append(f"{status} {test.name} — {test.message}")
    return "\n".join(lines)


def launch_visual_app(
    data: ProblemData,
    result: dict,
    tests: Sequence[TestResult] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button

    tests = list(tests) if tests is not None else []

    positions = compute_node_positions(data.oracle)
    current_result = result
    current_vehicle_count = len(current_result.get("routes", []))
    timelines = build_route_timelines(
        [route["sequence"] for route in current_result.get("routes", [])],
        data.oracle,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Simulation GTMS-Cert — clients et camions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.2)

    depot = data.depot_id
    depot_pos = positions[depot]
    client_nodes = [node for node in positions if node != depot]
    client_x = [positions[node][0] for node in client_nodes]
    client_y = [positions[node][1] for node in client_nodes]

    ax.scatter(client_x, client_y, s=25, c="#1f77b4", alpha=0.7, label="Clients")
    ax.scatter([depot_pos[0]], [depot_pos[1]], s=200, c="#d62728", marker="*", label="Dépôt", zorder=5)

    cmap_name = "tab20"
    route_lines: List[plt.Line2D] = []
    trucks_artists: List[tuple[RouteTimeline, tuple[float, float, float, float], plt.Line2D, plt.Line2D]] = []
    max_time = 0.0

    def _draw_routes() -> None:
        """Draw the route polylines and animated trucks for the current solution."""

        nonlocal trucks_artists, max_time, route_lines
        cmap = plt.cm.get_cmap(cmap_name, max(1, len(timelines)))
        max_time = 0.0
        trucks_artists = []
        route_lines = []
        for idx, timeline in enumerate(timelines):
            color = cmap(idx)
            xs = [positions[node][0] for node in timeline.route]
            ys = [positions[node][1] for node in timeline.route]
            route_line, = ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1.5)
            trail_line = ax.plot([], [], color=color, linewidth=2.0, alpha=0.8, zorder=6)[0]
            truck_point = ax.plot([], [], marker="s", color=color, markersize=8, zorder=10)[0]
            trucks_artists.append((timeline, color, truck_point, trail_line))
            route_lines.append(route_line)
            max_time = max(max_time, timeline.total_time)

    _draw_routes()

    def _format_stats(payload: dict, trucks: int) -> str:
        return (
            f"Camions : {trucks}\n"
            f"Makespan : {payload['makespan']:.2f} min\n"
            f"Gap : {payload['gap'] * 100:.2f}%"
        )

    stats_annotation = fig.text(
        0.02,
        0.95,
        _format_stats(current_result, current_vehicle_count),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#333333"),
    )

    summary = _format_test_summary(tests)
    if summary:
        fig.text(
            0.02,
            0.65,
            summary,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#333333"),
        )

    timer_text = ax.text(0.02, 0.02, "Temps écoulé : 0.0 min", transform=ax.transAxes)

    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")

    fps = 30
    base_interval = 1000 / fps
    speed_state = {"factor": 1.0}
    idle_tail = 5.0
    frames = int((max_time + idle_tail) * fps) + 1

    def interpolate_position(timeline: RouteTimeline, t: float) -> Coordinate:
        if not timeline.segments:
            return positions[timeline.route[0]]
        elapsed = 0.0
        for segment in timeline.segments:
            if t <= elapsed + segment.travel_time or segment == timeline.segments[-1]:
                start_pos = positions[segment.start]
                end_pos = positions[segment.end]
                if segment.travel_time <= 0:
                    return end_pos
                ratio = max(0.0, min(1.0, (t - elapsed) / segment.travel_time))
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio
                return (x, y)
            elapsed += segment.travel_time
        return positions[timeline.route[-1]]

    def compute_travel_path(timeline: RouteTimeline, t: float) -> List[Coordinate]:
        path: List[Coordinate] = [positions[timeline.route[0]]]
        if not timeline.segments:
            return path
        elapsed = 0.0
        for segment in timeline.segments:
            start_pos = positions[segment.start]
            end_pos = positions[segment.end]
            if t >= elapsed + segment.travel_time - 1e-9:
                path.append(end_pos)
                elapsed += segment.travel_time
                continue
            if segment.travel_time <= 0:
                path.append(end_pos)
            else:
                ratio = max(0.0, min(1.0, (t - elapsed) / segment.travel_time))
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio
                path.append((x, y))
            break
        return path

    def reset_trucks() -> None:
        for timeline, _color, point, trail in trucks_artists:
            start_x, start_y = positions[timeline.route[0]]
            point.set_data([start_x], [start_y])
            trail.set_data([start_x], [start_y])

    def update(frame_idx: int):
        current_time = frame_idx / fps
        display_time = min(current_time, max_time)
        artists = []
        for timeline, _color, point, trail in trucks_artists:
            x, y = interpolate_position(timeline, current_time)
            point.set_data([x], [y])
            path = compute_travel_path(timeline, current_time)
            xs = [coord[0] for coord in path]
            ys = [coord[1] for coord in path]
            trail.set_data(xs, ys)
            artists.extend([point, trail])
        timer_text.set_text(f"Temps écoulé : {display_time:.1f} min")
        return artists + [timer_text]

    anim: FuncAnimation | None = None

    speed_text = fig.text(
        0.75,
        0.12,
        "Vitesse x1.00",
        ha="left",
        va="center",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#333333"),
    )

    def _apply_speed() -> None:
        speed_text.set_text(f"Vitesse x{speed_state['factor']:.2f}")
        if anim is not None:
            event_source = getattr(anim, "event_source", None)
            if event_source is not None and speed_state["factor"] > 0:
                event_source.interval = base_interval / speed_state["factor"]
        fig.canvas.draw_idle()

    def _adjust_speed(delta: float) -> None:
        new_factor = max(0.25, min(4.0, speed_state["factor"] + delta))
        speed_state["factor"] = new_factor
        _apply_speed()

    def start_animation(_event) -> None:
        nonlocal anim
        if anim is not None:
            event_source = getattr(anim, "event_source", None)
            if event_source is not None:
                event_source.stop()
            anim = None
        reset_trucks()
        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=base_interval,
            repeat=False,
            blit=False,
        )
        _apply_speed()

    reset_trucks()

    proposal_text = fig.text(
        0.02,
        0.5,
        "",
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#555555"),
        visible=False,
    )

    proposal_solution: dict | None = None

    def _update_frames() -> None:
        nonlocal frames
        frames = int((max_time + idle_tail) * fps) + 1

    def _apply_solution(new_payload: dict, trucks: int) -> None:
        nonlocal current_result, current_vehicle_count, timelines
        current_result = new_payload
        current_vehicle_count = trucks
        for line in route_lines:
            line.remove()
        for _timeline, _color, point, trail in trucks_artists:
            point.remove()
            trail.remove()
        timelines = build_route_timelines(
            [route["sequence"] for route in current_result.get("routes", [])],
            data.oracle,
        )
        _draw_routes()
        _update_frames()
        reset_trucks()
        stats_annotation.set_text(_format_stats(current_result, current_vehicle_count))
        timer_text.set_text("Temps écoulé : 0.0 min")
        start_animation(None)
        fig.canvas.draw_idle()

    button_ax = fig.add_axes([0.75, 0.02, 0.2, 0.06])
    button = Button(button_ax, "Lancer la simulation", color="#4caf50", hovercolor="#66bb6a")
    button.on_clicked(start_animation)

    slow_ax = fig.add_axes([0.52, 0.02, 0.1, 0.06])
    slow_button = Button(slow_ax, "Ralentir", color="#f9a825", hovercolor="#ffca28")

    fast_ax = fig.add_axes([0.63, 0.02, 0.1, 0.06])
    fast_button = Button(fast_ax, "Accélérer", color="#1976d2", hovercolor="#42a5f5")

    propose_ax = fig.add_axes([0.31, 0.02, 0.18, 0.06])
    propose_button = Button(
        propose_ax,
        "Proposer gap 0",
        color="#8e24aa",
        hovercolor="#ba68c8",
    )

    apply_ax = fig.add_axes([0.12, 0.02, 0.18, 0.06])
    apply_button = Button(
        apply_ax,
        "Appliquer la proposition",
        color="#00695c",
        hovercolor="#26a69a",
    )

    def _on_slow(_event) -> None:
        _adjust_speed(-0.25)

    def _on_fast(_event) -> None:
        _adjust_speed(0.25)

    def _on_propose(_event) -> None:
        nonlocal proposal_solution
        proposal_text.set_visible(True)
        if current_result.get("gap", 0.0) <= 1e-6:
            proposal_text.set_text("Le gap est déjà nul : aucune amélioration nécessaire.")
            fig.canvas.draw_idle()
            return
        base_trucks = current_vehicle_count
        max_trucks_search = len(data.clients)
        best_payload: dict | None = None
        best_trucks = base_trucks
        try:
            for trucks in range(base_trucks + 1, max_trucks_search + 1):
                candidate_data = ProblemData(
                    depot_id=data.depot_id,
                    vehicle_count=trucks,
                    use_all=data.use_all,
                    clients=list(data.clients),
                    oracle=data.oracle,
                )
                routes, ub, lb, gap, longest_idx = solve_problem_data(
                    candidate_data,
                    seed=42,
                    lb_iters=150,
                )
                payload = build_output_payload(routes, data.oracle, ub, lb, gap, longest_idx)
                if gap <= 1e-6:
                    best_payload = payload
                    best_trucks = trucks
                    break
                if best_payload is None or gap < best_payload.get("gap", 1.0):
                    best_payload = payload
                    best_trucks = trucks
        except ValueError as exc:
            proposal_text.set_text(f"Erreur lors de la génération de la proposition : {exc}")
            proposal_solution = None
            fig.canvas.draw_idle()
            return

        if best_payload is None:
            proposal_text.set_text("Impossible de calculer une proposition améliorée.")
            proposal_solution = None
            fig.canvas.draw_idle()
            return

        extra = best_trucks - base_trucks
        gap_value = best_payload.get("gap", 0.0) * 100
        if gap_value <= 1e-4:
            proposal_text.set_text(
                f"Proposition : ajouter {extra} camion(s) (total {best_trucks}) pour atteindre un gap de 0 %."
            )
        else:
            proposal_text.set_text(
                (
                    "Gap 0 % non atteint. Meilleure suggestion : ajouter "
                    f"{extra} camion(s) (total {best_trucks}) pour un gap de {gap_value:.2f} %."
                )
            )
        proposal_solution = {"payload": best_payload, "trucks": best_trucks}
        fig.canvas.draw_idle()

    def _on_apply(_event) -> None:
        if proposal_solution is None:
            proposal_text.set_visible(True)
            proposal_text.set_text(
                "Aucune proposition disponible. Cliquez d'abord sur 'Proposer gap 0'."
            )
            fig.canvas.draw_idle()
            return
        selected = proposal_solution
        _apply_solution(selected["payload"], selected["trucks"])
        proposal_text.set_visible(True)
        proposal_text.set_text(
            (
                f"Solution appliquée avec {selected['trucks']} camions "
                f"(gap {selected['payload']['gap'] * 100:.2f} %)."
            )
        )
        proposal_solution = None
        fig.canvas.draw_idle()

    slow_button.on_clicked(_on_slow)
    fast_button.on_clicked(_on_fast)
    propose_button.on_clicked(_on_propose)
    apply_button.on_clicked(_on_apply)

    _apply_speed()

    start_animation(None)

    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualisation interactive du solveur GTMS-Cert")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Ne pas exécuter la suite de tests, uniquement la simulation",
    )
    parser.add_argument(
        "--cands",
        type=int,
        default=32,
        help="Nombre de voisins candidats pour l'oracle de distances",
    )
    args = parser.parse_args(argv)

    if args.skip_tests:
        payload_path = Path(__file__).resolve().parent / "tests" / "sample_instance_200_clients.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["vehicles"]["k"] = 10
        test_result, data, result = _run_scenario(
            "Simulation",
            payload,
            seed=0,
            cands=args.cands,
            lb_iters=150,
            expected_routes=10,
            expect_gap=0.01,
        )
        tests = [TestResult(name="Tests ignorés", passed=True, message="Non exécutés"), test_result]
    else:
        tests, data, result = run_regression_tests(args.cands)

    launch_visual_app(data, result, tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
