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

echo "==> Plotting from saved models (no retraining)..."
python main.py --plot-only --portfolio
echo ""
echo "==> Done! Charts saved to ./plot/"
echo "==> Press Enter to close..."
read
