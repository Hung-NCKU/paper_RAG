#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv-linux/bin/python ]; then
  python3 -m venv .venv-linux
fi
.venv-linux/bin/python -m pip install -r requirements.lock.txt
exec .venv-linux/bin/python -m uvicorn app.main:app --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}"
