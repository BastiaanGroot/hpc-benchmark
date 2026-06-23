#!/usr/bin/env bash
# Install MLPerf Storage v2.0 benchmark suite (mlcommons/storage)
# Run once per cluster environment.

set -euo pipefail

INSTALL_DIR="${MLPERF_STORAGE_DIR:-${HOME}/mlperf-storage}"

echo "Installing MLPerf Storage into ${INSTALL_DIR}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/mlcommons/storage.git "${INSTALL_DIR}"
else
  git -C "${INSTALL_DIR}" pull --ff-only
fi

cd "${INSTALL_DIR}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"

echo ""
echo "Installation complete."
echo "Activate with:  source ${INSTALL_DIR}/.venv/bin/activate"
echo "Run help with:  mlpstorage --help"
