#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Custom Wake Word - Full setup
# Run from the repo folder: bash setup.sh
#
# Prerequisites on the VM:
#   - Python 3.12
#   - CUDA 13.0 (torch 2.11.0+cu130)
#   - ~200 GB free disk space for datasets + venv + training
#
# What it does:
#   1. Checks Python 3.12 and creates the venv
#   2. pip install -r requirements.txt
#   3. 00_download.py - clones openwakeword + downloads datasets + Italian voice
#   4. 01_fix_n_patch.py - patches acoustics, torchaudio, openwakeword/train.py
#
# After:
#   Put your recordings in ./real_recordings/ and run:
#   python 02_training.py
# ─────────────────────────────────────────────────────────────────────────────
set -e

mkdir -p log
exec > >(tee -a log/setup.log) 2>&1

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

# Inject CUDA 12 lib path into venv activate so ORT finds libcublasLt.so.12
CUDA12_LIB="/usr/local/cuda-12.9/lib64"
if [ -d "$CUDA12_LIB" ] && ! grep -q "cuda-12.9" "$VENV_DIR/bin/activate"; then
    echo "export LD_LIBRARY_PATH=$CUDA12_LIB:\$LD_LIBRARY_PATH" >> "$VENV_DIR/bin/activate"
    echo "=== Injected $CUDA12_LIB into venv LD_LIBRARY_PATH ==="
    source "$VENV_DIR/bin/activate"
fi

# ── 3. System deps (cuBLAS for onnxruntime-gpu CUDA provider) ────────────────
echo ""
echo "=== System deps: CUDA 12 libs for onnxruntime CUDAExecutionProvider ==="
apt-get install -y \
    libcublas-12-9 \
    libcurand-12-9 \
    libcufft-12-9 \
    libcusolver-12-9 \
    libcusparse-12-9 \
    libcudnn9-cuda-12 \
    || echo "[WARN] apt install failed — CUDA provider may fall back to CPU"

# ── 4. pip install ────────────────────────────────────────────────────────────
echo ""
echo "=== pip install -r requirements.txt ==="
pip install --upgrade pip -q
pip install -r requirements.txt

# Reinstall onnxruntime-gpu from CUDA-12 feed (bundles cuDNN, needed for GPU inference)
echo ""
echo "=== onnxruntime-gpu: reinstall from CUDA-12 feed ==="
pip install onnxruntime-gpu \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    --force-reinstall --quiet

# ── 4. Clone repos + download datasets ───────────────────────────────────────
echo ""
echo "=== Download repos and datasets (00_download.py) ==="
echo "    (ACAV100M feature download is ~17 GB - this will take a while)"
python 00_download.py

# ── 6. Patch incompatible dependencies ───────────────────────────────────────
echo ""
echo "=== Patch incompatible dependencies (01_fix_n_patch.py) ==="
python 01_fix_n_patch.py

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
