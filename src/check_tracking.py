"""[client] Diagnose where MLflow runs from this shell would go.

    uv run python src/check_tracking.py        # exit 0 = server reachable, 2 = misconfigured

Prints the resolved MLFLOW_TRACKING_URI and whether a tracking server answers there.
It creates no run and writes nothing, so it never "proves" a local run reached the server.
"""

from __future__ import annotations

import sys

from tracking import diagnose


def main() -> int:
    ok, why = diagnose()
    print(("OK: " if ok else "PROBLEM: ") + why)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
