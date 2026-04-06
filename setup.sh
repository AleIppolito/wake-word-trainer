#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Custom Wake Word - Full setup
# Run from the repo folder: bash setup.sh
#
# Prerequisites on the VM:
#   - Python 3.12
#   - CUDA 13.0 (torch 2.10.0+cu130)
#   - ~20 GB free disk space for datasets + venv
#
# What it does:
#   1. Creates the venv (if it doesn't exist)
#   2. pip install -r requirements.txt
#   3. Applies patches to libraries (pronouncing, acoustics)
#   4. Clones piper-sample-generator and openwakeword
#   5. Downloads datasets and features (~17 GB — skips if already present)
#
# After:
#   Put your recordings in ./real_recordings/ and run:
#   python 02_training.py
# ─────────────────────────────────────────────────────────────────────────────
set -e

VENV_DIR="./.venv"
PYTHON="python3.12"

# ── 1. Check Python ───────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] $PYTHON not found. Install Python 3.12 and try again."
    exit 1
fi
echo "=== Python: $($PYTHON --version) ==="

# ── 2. Venv ───────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating venv at $VENV_DIR ==="
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "=== Venv already present: $VENV_DIR ==="
fi

source "$VENV_DIR/bin/activate"
echo "=== Venv activated: $(which python) ==="

# ── 3. pip install ────────────────────────────────────────────────────────────
echo ""
echo "=== pip install -r requirements.txt ==="
pip install --upgrade pip -q
pip install -r requirements.txt

# ── 4. Patch libraries ────────────────────────────────────────────────────────
echo ""
echo "=== Patch libraries (00_fix_dependencies.py) ==="
python 00_fix_dependencies.py

# ── 5. Clone repos + download datasets ───────────────────────────────────────
echo ""
echo "=== Setup repos and download datasets (01_setup_and_download.py) ==="
echo "    (ACAV100M feature download is ~17 GB — this will take a while)"
python 01_setup_and_download.py

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. Transfer your recordings to:"
echo "         ./real_recordings/   (WAV files, 16 kHz, ~300 clips)"
echo "       From your local machine:"
echo "         scp -r real_recordings/ root@<SERVER_IP>:$(pwd)/real_recordings"
echo ""
echo "    2. Start training:"
echo "         source .venv/bin/activate"
echo "         python 02_training.py"
echo "════════════════════════════════════════════════════════════"
