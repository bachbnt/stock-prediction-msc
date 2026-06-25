#!/bin/bash
# setup.sh — Create virtual environment and install dependencies
set -e
cd "$(dirname "$0")/.."

VENV_DIR="venv"

# ── 1. Find Python 3.10–3.13 (TensorFlow does not support 3.14+) ─────────────
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info[:2])")
        if [[ "$VER" == "(3, 13)" || "$VER" == "(3, 12)" || "$VER" == "(3, 11)" || "$VER" == "(3, 10)" ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "ERROR: Python 3.10–3.13 not found."
    echo "TensorFlow does not support Python 3.14+."
    echo ""
    echo "Install via pyenv:"
    echo "  brew install pyenv"
    echo "  pyenv install 3.13.0"
    echo "  pyenv local 3.13.0"
    echo "  bash scripts/setup.sh"
    exit 1
fi

echo "==> Using: $PYTHON ($($PYTHON --version))"

# ── 2. Create virtual environment ────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "==> Removing existing venv..."
    rm -rf "$VENV_DIR"
fi
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── 3. Install dependencies (excluding TensorFlow) ───────────────────────────
echo "==> Installing dependencies..."
pip install -q \
    "scikit-learn>=1.3.0" \
    "statsmodels>=0.14.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "matplotlib>=3.7.0" \
    "scipy>=1.11.0" \
    "requests>=2.31.0" \
    "beautifulsoup4>=4.12.0"

# vnstock is optional — skip silently if install fails
pip install -q "vnstock>=4.0.0" || echo "  [warn] vnstock install failed — skipping."

# ── 4. TensorFlow — platform-specific ────────────────────────────────────────
# tensorflow-metal is incompatible with TF 2.19+ and largely unmaintained.
# Pin TF to 2.18.x on Apple Silicon for stability; use latest on other platforms.
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "==> Apple Silicon detected — installing TensorFlow 2.18..."
    pip install -q "tensorflow==2.18.*"
else
    echo "==> Installing TensorFlow..."
    pip install -q "tensorflow>=2.13.0"
fi

echo ""
echo "Done! To run the project:"
echo "   double-click scripts/start.command"
echo "   or: source venv/bin/activate && python main.py"
