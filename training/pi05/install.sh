#!/usr/bin/env bash
# Install openpi (π₀.₅) pinned to a known-good commit.
# Run once per cluster environment or container build.
#
# Pinned commit: 15a9616 (Jun 16 2026) — last tested HEAD with π₀.₅ + PyTorch

set -euo pipefail

OPENPI_COMMIT="15a9616a00943ada6c20a0f158e3adb39df2ccac"
INSTALL_DIR="${OPENPI_DIR:-${HOME}/openpi}"

echo "Installing openpi @ ${OPENPI_COMMIT} into ${INSTALL_DIR}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "${INSTALL_DIR}"
fi

git -C "${INSTALL_DIR}" fetch origin
git -C "${INSTALL_DIR}" checkout "${OPENPI_COMMIT}"

python3 -m venv "${INSTALL_DIR}/.venv"
source "${INSTALL_DIR}/.venv/bin/activate"

pip install --upgrade pip uv
# openpi uses uv for fast installs
uv pip install -e "${INSTALL_DIR}[torch]"

# Upgrade PyTorch to cu128 for Blackwell (B200/B300 sm_100/sm_103) support
pip install --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# Patch transformers with openpi's custom Gemma/SigLIP/PaliGemma modifications
SITE_PKG=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cp -r "${INSTALL_DIR}/src/openpi/models_pytorch/transformers_replace/"* \
  "${SITE_PKG}/transformers/"

echo ""
echo "openpi installed."
echo "Activate:  source ${INSTALL_DIR}/.venv/bin/activate"
echo "Commit:    ${OPENPI_COMMIT}"
