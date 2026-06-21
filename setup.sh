#!/bin/bash
# setup.sh — Tạo virtual environment và cài dependencies
# Chạy 1 lần: bash setup.sh
# Sau đó mỗi lần dùng: source venv/bin/activate

set -e

VENV_DIR="venv"

echo "==> Python: $(python3 --version)"
echo "==> Tạo venv tại ./$VENV_DIR ..."
python3 -m venv $VENV_DIR

echo "==> Kích hoạt venv ..."
source $VENV_DIR/bin/activate

echo "==> Nâng cấp pip ..."
pip install --upgrade pip -q

echo "==> Cài dependencies ..."
pip install -r requirements.txt

# Apple Silicon: cài thêm tensorflow-metal để tận dụng GPU chip M-series
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "==> Apple Silicon phát hiện — cài tensorflow-metal (GPU acceleration) ..."
    pip install tensorflow-metal
fi

echo ""
echo "✓ Xong! Để chạy project:"
echo "   source venv/bin/activate"
echo "   python thesis.py"
