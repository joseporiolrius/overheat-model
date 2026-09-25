"""[client] Train one tracked overheat classifier run (S11).

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    uv run python src/train.py --c 1.0 --run-name baseline-c1
    for c in 0.001 0.01 0.1 10 100; do uv run python src/train.py --c $c; done
    uv run python src/train.py --c 1.0 --features temp_a,temp_b            # the deliberately bad model
    uv run python src/train.py --c 1.0 --class_weight balanced              # homework sweep

Each run logs, to experiment `overheat` on the tracking server:
  params   c, features, class_weight, seed, test_size, dataset_identity, dataset_seed, git_commit, git_dirty
  metrics  accuracy, f1, precision, recall (on the held-out 20% split)
  model    the fitted scikit-learn pipeline (artifact `model`), ready for `register.py`
  tags     baseline_run_id (the run named baseline-c1, when it exists) for homework comparisons
MLflow also records the source commit tag (mlflow.source.git.commit). Together with the seed and the
dataset identity, that commit is the reproducibility receipt (README §6.1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tracking import require_server

FEATURES = ("temp_a", "temp_b", "temp_diff")
DEFAULT_FEATURES = "temp_a,temp_b,temp_diff"
EXPERIMENT = "overheat"
BASELINE_RUN_NAME = "baseline-c1"


def parse_class_weight(value: str):
    """none | balanced | <w>, the weight of the overheat class (normal = 1)."""
    if value == "none":
        return None
    if value == "balanced":
        return "balanced"
    try:
        w = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--class_weight must be none, balanced or a number > 0, not {value!r}")
    if w <= 0:
        raise argparse.ArgumentTypeError("--class_weight number must be > 0")
    return {0: 1.0, 1: w}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--c", type=float, default=1.0, help="inverse regularization strength (> 0)")
    p.add_argument("--features", default=DEFAULT_FEATURES, help=f"comma list from {','.join(FEATURES)}")
    p.add_argument("--class_weight", default="none", help="none | balanced | weight of the overheat class")
    p.add_argument("--run-name", dest="run_name", help="default: c<c>-<features>-cw<class_weight>")
    p.add_argument("--seed", type=int, default=42, help="train/test split seed")
    p.add_argument("--data", default="data/readings.parquet")
    args = p.parse_args(argv)
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = [f for f in features if f not in FEATURES]
    if not features or unknown or len(set(features)) != len(features):
        p.error(f"--features must be a comma list of distinct names from {','.join(FEATURES)} "
                f"(got {args.features!r}{'; unknown: ' + ','.join(unknown) if unknown else ''})")
    if args.c <= 0:
        p.error("--c must be > 0")
    try:
        args.class_weight_value = parse_class_weight(args.class_weight)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    args.feature_list = features
    args.run_name = args.run_name or f"c{args.c:g}-{'+'.join(features)}-cw{args.class_weight}"
    return args


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def main(argv=None) -> int:
    args = parse_args(argv)
    require_server()                                    # exit 2 unless a tracking server answers

    data = Path(args.data)
    if not data.exists():
        print(f"ERROR: {data} not found. Generate it first: uv run python src/generate_data.py", file=sys.stderr)
        return 2
    meta = json.loads((pq.read_schema(data).metadata or {}).get(b"overheat_dataset", b"{}"))
    df = pd.read_parquet(data)

    import logging
    import warnings

    import mlflow
    import mlflow.sklearn

    # Two harmless MLflow notices would print on every run and hide the result line:
    # uv environments ship without pip, and the prediction output is an integer class.
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = df[args.feature_list], df["overheat"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=args.c, class_weight=args.class_weight_value, max_iter=1000))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred),
               "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred)}

    commit, dirty = git_state()
    mlflow.set_experiment(EXPERIMENT)
    baseline = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list", max_results=1,
                                  filter_string=f"attributes.run_name = '{BASELINE_RUN_NAME}'",
                                  order_by=["attributes.start_time ASC"])
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "c": args.c, "features": ",".join(args.feature_list), "class_weight": args.class_weight,
            "seed": args.seed, "test_size": 0.2,
            "dataset_identity": meta.get("dataset_identity", "unknown"), "dataset_seed": meta.get("seed", "unknown"),
            "git_commit": commit, "git_dirty": dirty,
        })
        mlflow.log_metrics(metrics)
        if baseline and baseline[0].info.run_id != run.info.run_id:
            mlflow.set_tag("baseline_run_id", baseline[0].info.run_id)
        mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(3))
    print(f"run {run.info.run_id} ({args.run_name}): accuracy {metrics['accuracy']:.4f}  f1 {metrics['f1']:.4f}  "
          f"features {','.join(args.feature_list)}  c {args.c:g}  class_weight {args.class_weight}")
    if dirty:
        print("WARNING: src/, pyproject.toml or uv.lock has uncommitted changes; commit them so the run's "
              "git_commit really reproduces it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
