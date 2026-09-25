"""Where does this client send runs? Shared guard for every lab script (S11).

The course tracking server runs on the VM at http://127.0.0.1:5000 (loopback only).
When MLFLOW_TRACKING_URI is unset, MLflow 3 silently falls back to a LOCAL store,
sqlite:///<current dir>/mlflow.db + ./mlruns. Run from another directory, that local
run never reaches the server ("invisible runs"). Run from the repo root, it half-writes
into the server's own mlflow.db and then fails on the first artifact. So the lab scripts
refuse to log anything unless the client points at an http(s) server that answers.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

COURSE_URI = "http://127.0.0.1:5000"


def diagnose(timeout: float = 3.0) -> tuple[bool, str]:
    """(ok, explanation) for the current MLFLOW_TRACKING_URI. Never contacts anything but that URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return False, ("MLFLOW_TRACKING_URI is not set. MLflow would log to a LOCAL "
                       f"sqlite:///{os.getcwd()}/mlflow.db + ./mlruns, NOT to your tracking server.\n"
                       f"  fix: export MLFLOW_TRACKING_URI={COURSE_URI}")
    if not uri.startswith(("http://", "https://")):
        return False, (f"MLFLOW_TRACKING_URI={uri} is a local store, not a tracking server; "
                       f"runs logged there will never appear in the server UI.\n"
                       f"  fix: export MLFLOW_TRACKING_URI={COURSE_URI}")
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=timeout) as resp:
            body = resp.read(64).decode(errors="replace").strip()
    except (urllib.error.URLError, OSError) as exc:
        return False, (f"MLFLOW_TRACKING_URI={uri} but no MLflow server answers there ({exc}).\n"
                       f"  fix: start the server (see README §2) or correct the host/port "
                       f"(course server: {COURSE_URI})")
    return True, f"MLFLOW_TRACKING_URI={uri} -> server healthy ({body})"


def require_server() -> str:
    """Exit 2 with a diagnosis unless the client points at a live tracking server."""
    ok, why = diagnose()
    if not ok:
        print(f"TRACKING NOT CONFIGURED: {why}", file=sys.stderr)
        sys.exit(2)
    return os.environ["MLFLOW_TRACKING_URI"]
