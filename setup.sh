#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Hey Murph - Setup completo
# Esegui dalla cartella del repo: bash setup.sh
#
# Prerequisiti sulla VM:
#   - Python 3.12
#   - CUDA 13.0 (torch 2.10.0+cu130)
#   - ~20 GB spazio libero per dataset + venv
#
# Cosa fa:
#   1. Crea il venv (se non esiste)
#   2. pip install -r requirements.txt
#   3. Applica patch alle librerie (pronouncing, acoustics)
#   4. Clona piper-sample-generator e openwakeword
#   5. Scarica dataset e feature (~17 GB — salta se già presenti)
#
# Dopo:
#   Metti le tue registrazioni in ./real_recordings/ e lancia:
#   python 02_training.py
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/venv"
PYTHON="python3.12"

cd "$REPO_DIR"

# ── 1. Controlla Python ───────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERRORE] $PYTHON non trovato. Installa Python 3.12 e riprova."
    exit 1
fi
echo "=== Python: $($PYTHON --version) ==="

# ── 2. Venv ───────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creo venv in $VENV_DIR ==="
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "=== Venv già presente: $VENV_DIR ==="
fi

source "$VENV_DIR/bin/activate"
echo "=== Venv attivato: $(which python) ==="

# ── 3. pip install ────────────────────────────────────────────────────────────
echo ""
echo "=== pip install -r requirements.txt ==="
pip install --upgrade pip --quiet
pip install -r requirements.txt

# ── 4. Patch librerie ─────────────────────────────────────────────────────────
echo ""
echo "=== Patch librerie (00_fix_dependencies.py) ==="
python 00_fix_dependencies.py

# ── 5. Clone repos + download dataset ────────────────────────────────────────
echo ""
echo "=== Setup repo e download dataset (01_setup_and_download.py) ==="
echo "    (il download delle feature ACAV100M è ~17 GB — pazienza)"
python 01_setup_and_download.py

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup completato."
echo ""
echo "  Prossimi step:"
echo "    1. Trasferisci le registrazioni nella cartella:"
echo "         ./real_recordings/   (file WAV 16kHz, ~300 clip)"
echo "       Comando dal tuo PC:"
echo "         scp -r hey_murph_recordings/ root@<IP_VM>:$(pwd)/real_recordings/"
echo ""
echo "    2. Avvia il training:"
echo "         source venv/bin/activate"
echo "         python 02_training.py"
echo "════════════════════════════════════════════════════════════"
