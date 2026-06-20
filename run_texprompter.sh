#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PIPELINE_API_ENABLED=1

echo "Starting TexPrompter backend API..."
python -m uvicorn services.api:app --host 127.0.0.1 --port 8000 > /dev/null 2>&1 &
API_PID=$!

cleanup() {
  echo "Shutting down TexPrompter backend API..."
  if kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" || true
  fi
}
trap cleanup EXIT

wait_for_api() {
  for i in $(seq 1 20); do
    if python - <<'PY'
import urllib.request
import urllib.error

try:
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as resp:
        if resp.status == 200:
            raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

if ! wait_for_api; then
  echo "ERROR: FastAPI backend did not start in time." >&2
  exit 1
fi

echo "FastAPI backend is ready at http://127.0.0.1:8000"

echo "Launching Streamlit UI..."
streamlit run app/streamlit_app.py
