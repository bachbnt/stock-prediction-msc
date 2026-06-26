#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate

# Check that models have been trained
if [ ! -d "models" ]; then
    echo "ERROR: No trained models found."
    echo "Run start.command first to train the models."
    echo ""
    echo "==> Press Enter to close..."
    read
    exit 1
fi

read -p "==> Forecast days (must match what you trained with) [default: 1]: " STEPS
STEPS=${STEPS:-1}

echo "==> Re-running portfolio optimization only..."
python main.py --portfolio-only --steps "$STEPS"

echo ""
echo "==> Done! Chart saved to ./plot/portfolio_optimization.png"
echo "==> Press Enter to close..."
read
