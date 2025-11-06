"""Utilities to supervise multiple GTMS-Cert solver runs in parallel.

This module targets a workstation with an 11th generation Intel i7 CPU and
16 GB of RAM. It starts a configurable batch of solver processes with
different seeds and monitors their progress to stop all runs once a target
optimality gap has been reached.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

DEFAULT_SEEDS = (21, 22, 23, 24, 25, 26)
DEFAULT_MAX_PROCS = 6
DEFAULT_TARGET_GAP = 1.0
DEFAULT_POLL_INTERVAL = 30
DEFAULT_TRUCKS = 5
DEFAULT_CLIENTS = 200


@dataclass
class SolverProcess:
    seed: int
    log_path: Path
    handle: subprocess.Popen[str]

    def terminate(self) -> None:
        if self.handle.poll() is None:
            try:
                self.handle.terminate()
            except ProcessLookupError:
                return

    def kill(self) -> None:
        if self.handle.poll() is None:
            try:
                self.handle.kill()
            except ProcessLookupError:
                return


class ParallelSupervisor:
    def __init__(
        self,
        solver_cmd: List[str],
        seeds: Iterable[int] = DEFAULT_SEEDS,
        max_procs: int = DEFAULT_MAX_PROCS,
        trucks: int = DEFAULT_TRUCKS,
        clients: int = DEFAULT_CLIENTS,
        extra_args: Iterable[str] | None = None,
        target_gap: float = DEFAULT_TARGET_GAP,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        runs_dir: Path | None = None,
    ) -> None:
        self.solver_cmd = solver_cmd
        self.seeds = list(seeds)
        self.max_procs = max_procs
        self.trucks = trucks
        self.clients = clients
        self.extra_args = list(extra_args or ())
        self.target_gap = target_gap
        self.poll_interval = poll_interval
        self.runs_dir = runs_dir or Path("runs")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.processes: list[SolverProcess] = []
        self.best_gap: float | None = None
        self.best_seed: int | None = None

    def build_command(self, seed: int) -> List[str]:
        return [
            *self.solver_cmd,
            "--trucks",
            str(self.trucks),
            "--clients",
            str(self.clients),
            "--seed",
            str(seed),
            *self.extra_args,
        ]

    def launch_solver(self, seed: int) -> SolverProcess:
        log_path = self.runs_dir / f"seed_{seed}.log"
        cmd = self.build_command(seed)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        solver_process = SolverProcess(seed=seed, log_path=log_path, handle=process)
        self.processes.append(solver_process)
        return solver_process

    def prune_finished(self) -> None:
        self.processes = [proc for proc in self.processes if proc.handle.poll() is None]

    def parse_gap_from_log(self, log_path: Path) -> float | None:
        if not log_path.exists():
            return None
        gap_value: float | None = None
        with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
            for line in log_file:
                if "Gap final" in line:
                    try:
                        gap_str = line.strip().split()[-1].rstrip("%")
                        gap_value = float(gap_str)
                    except (IndexError, ValueError):
                        continue
        return gap_value

    def update_best_gap(self) -> None:
        for proc in self.processes:
            gap = self.parse_gap_from_log(proc.log_path)
            if gap is None:
                continue
            if self.best_gap is None or gap < self.best_gap:
                self.best_gap = gap
                self.best_seed = proc.seed

    def reached_target(self) -> bool:
        return self.best_gap is not None and self.best_gap <= self.target_gap

    def terminate_all(self) -> None:
        for proc in self.processes:
            proc.terminate()
        deadline = time.time() + 5
        while time.time() < deadline:
            if all(proc.handle.poll() is not None for proc in self.processes):
                return
            time.sleep(0.2)
        for proc in self.processes:
            proc.kill()

    def run(self) -> None:
        pending_seeds = iter(self.seeds)
        try:
            while True:
                self.prune_finished()
                while len(self.processes) < self.max_procs:
                    try:
                        seed = next(pending_seeds)
                    except StopIteration:
                        break
                    solver_process = self.launch_solver(seed)
                    print(f"[INFO] Started solver seed {seed} -> {solver_process.log_path}")
                    time.sleep(1)
                if not self.processes:
                    break
                self.update_best_gap()
                if self.reached_target():
                    print(
                        f"[INFO] Target gap {self.target_gap:.2f}% reached by seed {self.best_seed} "
                        f"with best gap {self.best_gap:.2f}%."
                    )
                    break
                print(
                    f"[INFO] Active: {len(self.processes):02d} | Best gap: "
                    f"{self.best_gap if self.best_gap is not None else math.inf:.2f}%"
                )
                time.sleep(self.poll_interval)
        finally:
            self.terminate_all()
            for proc in self.processes:
                return_code = proc.handle.poll()
                print(
                    f"[INFO] Solver seed {proc.seed} finished with return code {return_code}."
                )
            if self.best_gap is not None:
                print(
                    f"[INFO] Best gap observed: {self.best_gap:.2f}% (seed {self.best_seed})."
                )
            else:
                print("[WARN] No gap value extracted from logs. Check log parsing patterns.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple GTMS-Cert solver instances in parallel.")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS, help="Seeds to run in parallel.")
    parser.add_argument("--max-procs", type=int, default=DEFAULT_MAX_PROCS, help="Maximum concurrent processes.")
    parser.add_argument("--target-gap", type=float, default=DEFAULT_TARGET_GAP, help="Stop once this gap (%%) is reached.")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Seconds between log scans.")
    parser.add_argument("--trucks", type=int, default=DEFAULT_TRUCKS, help="Trucks per solver run.")
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS, help="Clients per solver run.")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Directory to store solver log files."
    )
    parser.add_argument(
        "--solver-cmd",
        default="python -m gtms_cert.run_with_custom_trucks",
        help="Command used to launch the solver.",
    )
    parser.add_argument(
        "--solver-extra",
        nargs="*",
        default=["--no-save"],
        help="Extra arguments appended at the end of the solver command.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    solver_cmd = shlex.split(args.solver_cmd, posix=os.name != "nt")
    supervisor = ParallelSupervisor(
        solver_cmd=solver_cmd,
        seeds=args.seeds,
        max_procs=args.max_procs,
        trucks=args.trucks,
        clients=args.clients,
        extra_args=args.solver_extra,
        target_gap=args.target_gap,
        poll_interval=args.poll_interval,
        runs_dir=args.runs_dir,
    )
    supervisor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
