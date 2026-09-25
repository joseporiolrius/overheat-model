"""[client] Register a tracked run's model as the next version of a registered model (S11).

    uv run python src/register.py --run-id <RUN_ID> --name overheat-classifier

Checks that the run exists and has a logged `model`, registers THAT model, and prints the new
version with its source run. It exits 1 without creating a version if the run or its model is
missing, and 2 if the tracking server is not configured. The registry lives on the tracking
server (mlflow.db + mlartifacts/ on the VM), never in Git.
"""

from __future__ import annotations

import argparse
import sys

from tracking import require_server

DEFAULT_NAME = "overheat-classifier"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True, help="the source run (from train.py output or the UI)")
    p.add_argument("--name", default=DEFAULT_NAME, help=f"registered model name (default {DEFAULT_NAME})")
    args = p.parse_args(argv)
    require_server()

    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        run = client.get_run(args.run_id)
    except MlflowException:
        print(f"ERROR: run {args.run_id!r} does not exist on {mlflow.get_tracking_uri()}; nothing registered",
              file=sys.stderr)
        return 1
    models = mlflow.search_logged_models(experiment_ids=[run.info.experiment_id], output_format="list",
                                         filter_string=f"source_run_id = '{run.info.run_id}' AND name = 'model'")
    if not models:
        print(f"ERROR: run {run.info.run_id} ({run.info.run_name}) has no logged model 'model'; nothing registered",
              file=sys.stderr)
        return 1
    mv = mlflow.register_model(f"models:/{models[0].model_id}", args.name)
    mv = client.get_model_version(args.name, mv.version)
    if mv.run_id != run.info.run_id:
        print(f"ERROR: version {mv.version} links run {mv.run_id}, expected {run.info.run_id}", file=sys.stderr)
        return 1
    acc = run.data.metrics.get("accuracy")
    print(f"registered {args.name} version {mv.version} <- run {mv.run_id} ({run.info.run_name}), "
          f"model {models[0].model_id}, accuracy {acc:.4f}" if acc is not None else
          f"registered {args.name} version {mv.version} <- run {mv.run_id} ({run.info.run_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
