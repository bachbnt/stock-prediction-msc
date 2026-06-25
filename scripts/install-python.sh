#!/bin/bash
# install-python.sh — Install pyenv + Python 3.13 for this project
set -e

echo "==> Checking for Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "Homebrew is required. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

echo "==> Installing pyenv..."
brew install pyenv

# Add pyenv to shell config if not already present
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "pyenv init" "$SHELL_RC"; then
        echo '' >> "$SHELL_RC"
        echo '# pyenv' >> "$SHELL_RC"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
        echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"
        echo "==> Added pyenv to $SHELL_RC"
    fi
fi

# Load pyenv in the current shell session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "==> Installing Python 3.13.0 (this may take 5–10 minutes)..."
pyenv install 3.13.0

echo "==> Setting Python 3.13.0 as the local version for this project..."
cd "$(dirname "$0")/.."
pyenv local 3.13.0

echo ""
echo "Done! Now run:"
echo "   bash scripts/setup.sh"
