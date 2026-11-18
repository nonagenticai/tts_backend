#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${TTS_PORT:-9202}"
ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/tts-backend/env"
if [ ! -f "$ENV_FILE" ] && [ -f /etc/tts-backend/env ]; then ENV_FILE=/etc/tts-backend/env; fi
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
lsof -ti :"${PORT}" | xargs kill -9 2>/dev/null || true
sleep 1
mkdir -p logs
export SSL_CERT_FILE="${SSL_CERT_FILE:-/opt/homebrew/etc/ca-certificates/cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/opt/homebrew/etc/ca-certificates/cert.pem}"
echo "Starting TTS Backend on port ${PORT}..."
PYTHONPATH=src exec uv run uvicorn main:app --host 0.0.0.0 --port "${PORT}"
