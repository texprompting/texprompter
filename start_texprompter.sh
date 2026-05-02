#!/usr/bin/env bash
# Dev launcher (Unix): SSH tunnel + MLflow + interactive menu. See README «Dev startup scripts».
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[dev] Tip: conda activate texprompting if python/mlflow imports fail." >&2
python scripts/texprompter_dev.py "$@"
