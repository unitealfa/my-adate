"""Compatibility wrapper to expose the VRP solver package.

Historically the implementation lives in ``Cluster‑First/solver``.  When the
project is executed from the repository root (e.g. via ``pytest``) Python does
not automatically discover that package because the directory ``Cluster‑First``
contains a hyphen and therefore is not on ``sys.path`` by default.  This module
bridges the gap by forwarding the package search path to the canonical
implementation directory.
"""
from __future__ import annotations

from pathlib import Path

# Resolve the location of the actual implementation.
_CLUSTER_FIRST_DIR = Path(__file__).resolve().parent.parent / "Cluster‑First"
_SOLVER_DIR = _CLUSTER_FIRST_DIR / "solver"

if not _SOLVER_DIR.is_dir():  # pragma: no cover - defensive programming
    raise ImportError(
        "Cannot locate the solver package inside 'Cluster‑First/solver'. Ensure "
        "the project structure is intact."
    )

# Update the package search path so ``import solver.data`` loads the modules
# hosted in ``Cluster‑First/solver``.
__path__ = [str(_SOLVER_DIR)]

# Keep the import machinery aware of the redirected location.  This mirrors what
# Python would normally populate for a package.
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__
