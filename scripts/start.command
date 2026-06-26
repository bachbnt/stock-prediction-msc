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

VALID_SYMBOLS="FPT VCB VIC HPG VNM MWG SSI PNJ"

echo "─────────────────────────────────────────"
echo "  Stock Prediction — Configuration"
echo "  Valid symbols: $VALID_SYMBOLS"
echo "─────────────────────────────────────────"

# ── Symbols (with validation) ─────────────────────────────────────────────────
while true; do
    read -p "  Symbols to run       [default: all]:     " SYMBOLS_INPUT
    if [ -z "$SYMBOLS_INPUT" ]; then
        break  # use all
    fi
    INVALID=""
    for sym in $SYMBOLS_INPUT; do
        SYM_UPPER=$(echo "$sym" | tr '[:lower:]' '[:upper:]')
        if ! echo "$VALID_SYMBOLS" | grep -qw "$SYM_UPPER"; then
            INVALID="$INVALID $SYM_UPPER"
        fi
    done
    if [ -z "$INVALID" ]; then
        SYMBOLS_INPUT=$(echo "$SYMBOLS_INPUT" | tr '[:lower:]' '[:upper:]')
        break
    else
        echo "  ERROR: Unknown symbol(s):$INVALID. Please try again."
    fi
done

# ── Steps (must be a positive integer) ────────────────────────────────────────
while true; do
    read -p "  Forecast days ahead  [default: 1]:       " STEPS
    STEPS=${STEPS:-1}
    if [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
        break
    else
        echo "  ERROR: Please enter a positive integer (e.g. 1, 5, 10)."
    fi
done

read -p "  Refresh data first?  [y/N]:              " REFRESH
read -p "  Include sentiment?   [y/N]:              " SENTIMENT

echo "─────────────────────────────────────────"
echo ""

# ── Build and run command ──────────────────────────────────────────────────────
CMD="python main.py --portfolio --steps $STEPS"

[ -n "$SYMBOLS_INPUT" ]          && CMD="$CMD --symbols $SYMBOLS_INPUT"
[[ "$REFRESH"   =~ ^[Yy]$ ]]    && CMD="$CMD --refresh"
[[ "$SENTIMENT" =~ ^[Yy]$ ]]    && CMD="$CMD --sentiment"

echo "==> Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "==> Done! Charts saved to ./plot/"
echo "==> Press Enter to close..."
read
