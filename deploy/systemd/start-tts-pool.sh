#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/gunnar/projects/tts-pool"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
VENV_BIN="$ROOT_DIR/.venv/bin"
SETTINGS_PATH="${TTS_POOL_SETTINGS_PATH:-$ROOT_DIR/config/settings.json}"
HOST="${HOST:-127.0.0.1}"
DEFAULT_PORT="${DEFAULT_PORT:-8020}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python venv: $PYTHON_BIN" >&2
  exit 127
fi

export PATH="$VENV_BIN:$PATH"

PORT="$DEFAULT_PORT"
if [[ -f "$SETTINGS_PATH" ]]; then
  SETTINGS_PORT="$("$PYTHON_BIN" -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); payload=json.loads(p.read_text(encoding='utf-8')); s=payload.get('service',{}) if isinstance(payload,dict) else {}; print(s.get('port','') if isinstance(s,dict) else '')" "$SETTINGS_PATH" 2>/dev/null || true)"
  if [[ -n "$SETTINGS_PORT" ]]; then
    PORT="$SETTINGS_PORT"
  fi
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
