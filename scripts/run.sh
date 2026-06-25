#!/bin/bash
# run.sh — Run the full stock price prediction pipeline
#
# Usage:
#   bash run.sh                      # default run (5 symbols, single-step)
#   bash run.sh --refresh            # update data from vnstock first
#   bash run.sh --steps 5            # multi-step forecast (5 days)
#   bash run.sh --portfolio          # include Markowitz optimization
#   bash run.sh --symbols FPT VCB    # run 2 symbols only
#   bash run.sh --refresh --portfolio --steps 5   # combine flags

set -e
cd "$(dirname "$0")/.."

VENV_DIR="venv"

# ── 1. Create venv if it doesn't exist ────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "==> No venv found, running first-time setup..."
    bash scripts/setup.sh
fi

# ── 2. Activate venv ──────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "==> Python: $(python --version)"

# ── 3. Run pipeline ───────────────────────────────────────────────────────────
echo "==> Starting pipeline... (args: $@)"
echo ""
python main.py "$@"

echo ""
echo "==> Done! Charts saved to ./plot/"
