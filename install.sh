#!/usr/bin/env bash
# install.sh — one-click Python dependency installer for inference-modeling.
#
# Usage:
#   ./install.sh                # create ./myenv venv (default) and install
#   ./install.sh --user         # install into the active Python's user site
#                               # (skips venv creation)
#   ./install.sh --system       # install into the active Python globally
#                               # (may need sudo / pip --break-system-packages)
#   ./install.sh --venv PATH    # create / reuse venv at PATH instead of ./myenv
#   ./install.sh -h | --help    # show this help
#
# After a venv install, activate with:
#   source <venv>/bin/activate          # bash / zsh
#   source <venv>/bin/activate.fish     # fish
#
# Then sanity-check with:
#   python3 -m simulator --gpu H100 --model deepseek-v3.2 \
#       --tp 4 --ep 32 --dp 8 --batch 64 --ctx 131072
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

MODE="venv"
VENV_DIR="${SCRIPT_DIR}/myenv"
PYTHON_BIN="${PYTHON:-python3}"

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)    MODE="user";    shift ;;
        --system)  MODE="system";  shift ;;
        --venv)    MODE="venv"; VENV_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
    esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[install.sh] ERROR: ${PYTHON_BIN} not found in PATH." >&2
    echo "             Install Python 3.9+ or set PYTHON=/path/to/python3." >&2
    exit 1
fi

PY_VERSION="$(${PYTHON_BIN} -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "[install.sh] Using ${PYTHON_BIN} (Python ${PY_VERSION})"
${PYTHON_BIN} -c 'import sys;sys.exit(0 if sys.version_info >= (3,9) else 1)' \
    || { echo "[install.sh] ERROR: Python >= 3.9 required, found ${PY_VERSION}." >&2; exit 1; }

case "${MODE}" in
    venv)
        if [[ ! -d "${VENV_DIR}" ]]; then
            echo "[install.sh] Creating venv at ${VENV_DIR}"
            "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        else
            echo "[install.sh] Reusing existing venv at ${VENV_DIR}"
        fi
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        PIP=(python -m pip)
        ;;
    user)
        echo "[install.sh] Installing into user site (--user)"
        PIP=("${PYTHON_BIN}" -m pip install --user)
        # rebind PIP to upgrade form below
        PIP_INSTALL=("${PYTHON_BIN}" -m pip install --user)
        ;;
    system)
        echo "[install.sh] Installing into system Python (no venv)"
        PIP=("${PYTHON_BIN}" -m pip)
        ;;
esac

if [[ "${MODE}" == "user" ]]; then
    "${PYTHON_BIN}" -m pip install --user --upgrade pip >/dev/null
    "${PYTHON_BIN}" -m pip install --user -r "${SCRIPT_DIR}/requirements.txt"
else
    "${PIP[@]}" install --upgrade pip >/dev/null
    "${PIP[@]}" install -r "${SCRIPT_DIR}/requirements.txt"
fi

echo
echo "[install.sh] Verifying imports..."
if [[ "${MODE}" == "venv" ]]; then
    python -c 'import numpy, yaml; print(f"  numpy  {numpy.__version__}"); print(f"  pyyaml {yaml.__version__}")'
    python -c 'import simulator; print("  simulator OK (package importable)")'
    echo
    echo "[install.sh] Done. Activate the venv with:"
    echo "    source ${VENV_DIR}/bin/activate"
else
    "${PYTHON_BIN}" -c 'import numpy, yaml; print(f"  numpy  {numpy.__version__}"); print(f"  pyyaml {yaml.__version__}")'
    "${PYTHON_BIN}" -c 'import simulator; print("  simulator OK (package importable)")'
    echo
    echo "[install.sh] Done."
fi
