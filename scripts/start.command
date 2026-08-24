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
DEFAULT_SYMBOLS="FPT VCB VIC HPG VNM"
DEFAULT_STEPS=1

echo "─────────────────────────────────────────"
echo "  Stock Prediction — Configuration"
echo "  Valid symbols: $VALID_SYMBOLS"
echo "─────────────────────────────────────────"

# ── Symbols (with validation) ─────────────────────────────────────────────────
while true; do
    read -p "  Symbols to run       [default: $DEFAULT_SYMBOLS]: " SYMBOLS_INPUT
    if [ -z "$SYMBOLS_INPUT" ]; then
        SYMBOLS_INPUT="$DEFAULT_SYMBOLS"
        break
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
    read -p "  Forecast days ahead  [default: $DEFAULT_STEPS]:       " STEPS
    STEPS=${STEPS:-$DEFAULT_STEPS}
    if [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
        break
    else
        echo "  ERROR: Please enter a positive integer (e.g. 1, 5, 10)."
    fi
done

read -p "  Refresh data first?  [default: No] (y/N): " REFRESH
read -p "  Include sentiment?   [default: No] (y/N): " SENTIMENT

REFRESH=${REFRESH:-N}
SENTIMENT=${SENTIMENT:-N}
REFRESH_LABEL="No"
SENTIMENT_LABEL="No"
[[ "$REFRESH"   =~ ^[Yy]$ ]] && REFRESH_LABEL="Yes"
[[ "$SENTIMENT" =~ ^[Yy]$ ]] && SENTIMENT_LABEL="Yes"

echo "─────────────────────────────────────────"
echo "  Selected configuration"
echo "  Symbols:             $SYMBOLS_INPUT"
echo "  Forecast days:       $STEPS"
echo "  Portfolio:           Yes"
echo "  Refresh data:        $REFRESH_LABEL"
echo "  Include sentiment:   $SENTIMENT_LABEL"
echo "─────────────────────────────────────────"
echo ""

# ── Build and run command ──────────────────────────────────────────────────────
CMD=(python main.py --portfolio --steps "$STEPS" --symbols)
for sym in $SYMBOLS_INPUT; do
    CMD+=("$sym")
done

[[ "$REFRESH"   =~ ^[Yy]$ ]] && CMD+=(--refresh)
[[ "$SENTIMENT" =~ ^[Yy]$ ]] && CMD+=(--sentiment)

printf "==> Running:"
printf " %q" "${CMD[@]}"
echo ""
echo ""
"${CMD[@]}"

echo ""
echo "==> Done! Charts saved to ./plot/"
echo "==> Press Enter to close..."
read
