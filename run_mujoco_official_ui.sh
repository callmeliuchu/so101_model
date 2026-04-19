#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMULATE_BIN="${SCRIPT_DIR}/.official-mujoco/MuJoCo.app/Contents/MacOS/simulate"
MODEL_NAME="${1:-task_scene}"
MODEL_PATH="${SCRIPT_DIR}/${MODEL_NAME}.xml"

if [[ ! -x "${SIMULATE_BIN}" ]]; then
  echo "Missing official MuJoCo simulate binary: ${SIMULATE_BIN}" >&2
  echo "Expected app bundle under ${SCRIPT_DIR}/.official-mujoco/MuJoCo.app" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  echo "Usage: $0 [scene|task_scene|so101_new_calib|so101_old_calib|task_scene_old]" >&2
  exit 1
fi

exec "${SIMULATE_BIN}" "${MODEL_PATH}"
