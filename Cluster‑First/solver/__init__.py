"""Expose the VRP solver package.

This module lives inside ``Cluster-First/solver`` and needs no special path
manipulation beyond pointing ``__path__`` to the current directory.  Keeping the
logic explicit avoids brittle relative-path computations when the project is
copied or renamed.
"""
from __future__ import annotations

from pathlib import Path

_SOLVER_DIR = Path(__file__).resolve().parent

# Update the package search path so ``import solver.data`` loads the modules
# hosted in ``Cluster-First/solver`` regardless of the working directory used to
# launch the interpreter.
__path__ = [str(_SOLVER_DIR)]

if __spec__ is not None:
    __spec__.submodule_search_locations = __path__