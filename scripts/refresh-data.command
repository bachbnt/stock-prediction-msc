#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate
echo "==> Updating data from vnstock (no model training)..."
python main.py --data-only
echo ""
echo "==> Press Enter to close..."
read
