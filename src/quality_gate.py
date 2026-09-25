"""[client] Quality gate: does a registered model version meet the accuracy bar? (S11)

    uv run python src/quality_gate.py --name overheat-classifier --version 1 --min-accuracy 0.85
    echo $?        # 0 = PASSED, 1 = REJECTED, 2 = cannot evaluate (never a pass)

The gate reads `accuracy` from the run linked to exactly that model version, then prints the
measured value and the threshold. Accuracy equal to the threshold passes. The exit code is the whole
interface: a CI step (S12) fails on any non-zero code, and the pipeline stops before promotion.
"""

from __future__ import annotations

import argparse
import sys

from tracking import require_server


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", default="overheat-classifier")
    p.add_argument("--version", required=True, help="registered model version to gate, e.g. 1")
    p.add_argument("--min-accuracy", type=float, default=0.85)
    args = p.parse_args(argv)
    if not 0.0 <= args.min_accuracy <= 1.0:
        p.error("--min-accuracy must be between 0 and 1")
    require_server()                                          # no server -> exit 2, never a pass

    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        run = client.get_run(client.get_model_version(args.name, args.version).run_id)
    except MlflowException as exc:
        print(f"CANNOT EVALUATE: {args.name} version {args.version} or its source run was not found "
              f"({exc.error_code})", file=sys.stderr)
        return 2
    acc = run.data.metrics.get("accuracy")
    if acc is None:
        print(f"CANNOT EVALUATE: run {run.info.run_id} of {args.name} v{args.version} has no 'accuracy' metric",
              file=sys.stderr)
        return 2
    shown = f"{acc:.4f}" if abs(acc - args.min_accuracy) >= 1e-4 else repr(acc)   # full precision at the edge
    if acc < args.min_accuracy:
        print(f"REJECTED: {args.name} v{args.version} accuracy {shown} < {args.min_accuracy} "
              f"(run {run.info.run_name})")
        return 1
    print(f"PASSED: {args.name} v{args.version} accuracy {shown} >= {args.min_accuracy} (run {run.info.run_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
