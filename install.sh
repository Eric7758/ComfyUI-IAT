#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

REQ="$(pwd)/requirements.txt"
if [[ ! -f "$REQ" ]]; then
  echo "[IAT] requirements.txt not found, skipping."
  exit 0
fi

PYTHON_EXE="python3"
if [[ -f "../../python_embeded/bin/python3" ]]; then
  PYTHON_EXE="../../python_embeded/bin/python3"
elif [[ -f "../../python/bin/python3" ]]; then
  PYTHON_EXE="../../python/bin/python3"
fi

echo "[IAT] Using Python: $PYTHON_EXE"
"$PYTHON_EXE" install.py
echo "[IAT] Done."
