from __future__ import annotations

import sys
from pathlib import Path

# Ensure the Cluster-First package directory (which contains the ``solver``
# package used across the tests) is available on ``sys.path`` when the test
# suite is executed from the repository root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))