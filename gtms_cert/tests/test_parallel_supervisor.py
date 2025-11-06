import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gtms_cert.parallel_supervisor import ParallelSupervisor


def test_build_command_without_problem_injection(tmp_path):
    supervisor = ParallelSupervisor(
        solver_cmd=["python", "-m", "dummy"],
        seeds=[99],
        max_procs=1,
        trucks=None,
        clients=None,
        extra_args=["--foo", "bar"],
        runs_dir=Path(tmp_path),
    )

    command = supervisor.build_command(99)

    assert command == ["python", "-m", "dummy", "--seed", "99", "--foo", "bar"]
