#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
CONDA_BIN=""
PYTHON_BIN="${ENV_PREFIX}/bin/python"

for candidate in "/Users/liuchu/miniconda3/bin/conda" "$(command -v conda 2>/dev/null || true)"; do
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    CONDA_BIN="${candidate}"
    break
  fi
done

if [[ -z "${CONDA_BIN}" ]]; then
  echo "usable conda not found" >&2
  exit 1
fi

if [[ -d "${ENV_PREFIX}" ]]; then
  echo "Updating local conda env at ${ENV_PREFIX}"
  "${CONDA_BIN}" env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}" --prune
else
  echo "Creating local conda env at ${ENV_PREFIX}"
  "${CONDA_BIN}" env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
  "mujoco==3.3.6" \
  "glfw>=2.7,<3" \
  "numpy>=1.26,<3" \
  "imageio>=2.34,<3" \
  "imageio-ffmpeg>=0.5,<1" \
  "opencv-python>=4.10,<5" \
  "mediapipe>=0.10.14,<1" \
  "pyarrow>=17,<24"

echo ""
echo "Environment ready."
echo "Run the viewer with:"
echo "  ${SCRIPT_DIR}/run_mujoco.sh"
