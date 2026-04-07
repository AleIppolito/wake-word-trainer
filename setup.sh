#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Custom Wake Word - Full setup
# Run from the repo folder: bash setup.sh
#
# Prerequisites on the VM:
#   - Python 3.12
#   - CUDA 13.0 (torch 2.10.0+cu130)
#   - ~200 GB free disk space for datasets + venv + training
#
# What it does:
#   1. Checks Python 3.12 and creates the venv
#   2. pip install -r requirements.txt (--no-deps) + torchcodec
#   3. 00_download.py - clones repos + downloads datasets (~17 GB)
#   4. 01_fix_n_patch.py - patches pronouncing and acoustics
#   5. Uninstalls torchcodec (conflicts with pinned dependencies)
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
pip install --no-deps -r requirements.txt

# ── 4. Install torchcodec (needed by datasets>=3.x for audio decoding) ───────
echo ""
echo "=== pip install torchcodec (temporary - removed after download) ==="
pip install torchcodec

# ── 5. Clone repos + download datasets ───────────────────────────────────────
echo ""
echo "=== Download repos and datasets (00_download.py) ==="
echo "    (ACAV100M feature download is ~17 GB - this will take a while)"
python 00_download.py

# ── 6. Patch incompatible dependencies ───────────────────────────────────────
echo ""
echo "=== Patch incompatible dependencies (01_fix_n_patch.py) ==="
python 01_fix_n_patch.py

# ── 7. Remove torchcodec (conflicts with pinned dependencies) ─────────────────
echo ""
echo "=== Removing torchcodec ==="
pip uninstall -y torchcodec

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
