#!/usr/bin/env bash
# [server] FALLBACK ONLY: run the course-mirrored MLflow image instead of `uv run mlflow server`.
#
#   ./scripts/mlflow-fallback.sh            # foreground; Ctrl-C stops it (README §10)
#
# Same data, same URL: it serves THIS directory's mlflow.db and mlartifacts/ on 127.0.0.1:5000,
# so the SSH tunnel and MLFLOW_TRACKING_URI stay unchanged. It refuses to start while anything
# (for example the normal uv server) already listens on 127.0.0.1:5000: two servers must never
# write the same mlflow.db. The image is pinned by digest (course-artifacts.yaml).
set -euo pipefail

IMAGE="ghcr.io/oriolrius/mlflow:v3.16.1@sha256:06058cc872276873e9759c635e773342aca4731e240f6d20d88ea43774083984"
PORT=5000

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] && grep -q '^name = "overheat-model"' pyproject.toml \
  || { echo "FALLBACK REFUSED: run it from your overheat-model repository" >&2; exit 2; }
command -v docker >/dev/null || { echo "FALLBACK REFUSED: docker is not installed on this VM" >&2; exit 2; }
if ss -ltn "( sport = :$PORT )" | grep -q ":$PORT"; then
  echo "FALLBACK REFUSED: 127.0.0.1:$PORT is already in use (is the uv MLflow server still running?)." >&2
  echo "  Stop it first (Ctrl-C in its [server] session). Two servers must never share mlflow.db." >&2
  exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "pulling $IMAGE ..."
  docker pull -q "$IMAGE" >/dev/null || { echo "FALLBACK FAILED: cannot pull $IMAGE (network? registry?)" >&2; exit 1; }
fi
echo "MLflow fallback (container) serving $(pwd)/mlflow.db + mlartifacts/ on http://127.0.0.1:$PORT (Ctrl-C to stop)"
exec docker run --rm --name overheat-mlflow-fallback \
  -p "127.0.0.1:$PORT:5000" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$(pwd):/work" -w /work "$IMAGE" \
  mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
