#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.conda/bin/python"
GLFW_LIB_DIR=""

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing local env: ${PYTHON_BIN}" >&2
  echo "Run ${SCRIPT_DIR}/setup_conda_env.sh first." >&2
  exit 1
fi

for candidate in "/usr/local/lib" "/opt/homebrew/lib"; do
  if [[ -f "${candidate}/libglfw.dylib" ]]; then
    GLFW_LIB_DIR="${candidate}"
    break
  fi
done

if [[ -n "${GLFW_LIB_DIR}" ]]; then
  export DYLD_LIBRARY_PATH="${GLFW_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/webcam_dual_teleop.py" "$@"
