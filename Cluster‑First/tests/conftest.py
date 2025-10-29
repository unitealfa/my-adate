from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (which contains the compatibility ``solver`` package)
# is present on ``sys.path`` when the tests are executed via ``pytest``.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))