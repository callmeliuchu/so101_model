#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.conda/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing local env: ${PYTHON_BIN}" >&2
  echo "Run ${SCRIPT_DIR}/setup_conda_env.sh first." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/import_hf_lerobot_episode.py" "$@"
