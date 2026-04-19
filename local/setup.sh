#!/usr/bin/env bash
# Local setup — creates venv and installs recorder deps.
# Run from the repo root: bash local/setup.sh
set -e

VENV_DIR="local/.venv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating venv at $VENV_DIR ==="
    python3 -m venv "$VENV_DIR"
else
    echo "=== Venv already present: $VENV_DIR ==="
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r local/requirements.txt

echo ""
echo "════════════════════════════════════════════════"
echo "  Setup complete. To record:"
echo ""
echo "    source local/.venv/bin/activate"
echo "    python local/record.py --mode close --room cucina"
echo ""
echo "  Auto-loop (hands-free):"
echo "    python local/record.py --mode far --room sala --auto"
echo ""
echo "  Validate existing clips:"
echo "    python local/record.py --validate"
echo "════════════════════════════════════════════════"
