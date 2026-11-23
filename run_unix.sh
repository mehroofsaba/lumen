#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
python main.py "$@"
