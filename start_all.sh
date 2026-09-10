#!/bin/bash
# Starts all SignSight services using the project venv (.venv-ml), which holds
# both the web-service deps and the ML deps the vision service needs to load
# the trained model. Create it first if missing:
#   python3.11 -m venv .venv-ml && source .venv-ml/bin/activate && pip install -r requirements.txt
cd "$(dirname "$0")"
PYTHON="$PWD/.venv-ml/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Error: .venv-ml not found. See comment at the top of this script." >&2
  exit 1
fi
echo "Starting SignSight Microservices..."
(cd gateway && "$PYTHON" -m uvicorn main:app --port 8000 --reload) &
(cd preprocessing-service && "$PYTHON" -m uvicorn app:app --port 8001 --reload) &
(cd vision-service && "$PYTHON" -m uvicorn app:app --port 8002 --reload) &
(cd nlp-service && "$PYTHON" -m uvicorn app:app --port 8003 --reload) &
# The frontend MUST be served over http://localhost, not opened as a file://
# URL: getUserMedia only runs in a secure context, and file:// is not one, so
# double-clicking index.html silently yields a camera-less page.
(cd frontend && "$PYTHON" -m http.server 8080) &
sleep 2
echo ""
echo "All services running. Open:  http://localhost:8080/index.html"
echo "(do NOT open frontend/index.html directly — file:// blocks the camera)"
