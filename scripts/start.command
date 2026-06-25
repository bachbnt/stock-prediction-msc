#!/bin/bash
cd "$(dirname "$0")/.."

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> First run: setting up virtual environment..."
    bash scripts/setup.sh
fi

source "$VENV_DIR/bin/activate"
echo "==> Python: $(python --version)"
echo ""

python main.py --portfolio

echo ""
echo "==> Done! Charts saved to ./plot/"
echo "==> Press Enter to close..."
read
