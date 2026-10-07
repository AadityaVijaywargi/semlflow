#!/usr/bin/env bash
# Run any command with MLflow tracking enabled.
#   * If MLFLOW_TRACKING_URI is already set (remote server / DagsHub) it is used as-is.
#   * Otherwise a local MLflow *server* is started (sqlite + served artifacts), the command
#     runs against it, and the server is stopped afterwards.
# Using a server (instead of a bare file store) makes artifact paths portable, so the
# resulting mlflow.db + mlartifacts/ can be baked into a Docker image (see Dockerfile.ui).
#
# Usage:  bash scripts/run_with_mlflow.sh dvc repro
set -euo pipefail

if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  echo ">> Using external MLflow: ${MLFLOW_TRACKING_URI}"
  exec "$@"
fi

PORT="${MLFLOW_PORT:-5000}"
mlflow server --host 127.0.0.1 --port "$PORT" \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts > mlflow-server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null && break
  sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || { cat mlflow-server.log; exit 1; }

export MLFLOW_TRACKING_URI="http://127.0.0.1:${PORT}"
echo ">> MLflow server up at ${MLFLOW_TRACKING_URI}"
"$@"
