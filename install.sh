#!/usr/bin/env sh
# Install Viola into its own virtualenv, put `viola` on PATH and download the classifier model.
#
#   ./install.sh                                 # from a clone of the repo
#   VIOLA_SRC=git+https://<repo-url> sh install.sh
#
# CPU-only PyTorch is installed on Linux and Windows (no CUDA download); macOS uses the default wheel.
set -eu

VIOLA_HOME="${VIOLA_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/viola}"
BIN_DIR="${VIOLA_BIN_DIR:-$HOME/.local/bin}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${VIOLA_SRC:-}" ]; then
  if [ -f "$HERE/pyproject.toml" ]; then VIOLA_SRC="$HERE"; else
    echo "Set VIOLA_SRC to the viola repo (path or git+https URL)." >&2; exit 1
  fi
fi

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Viola needs Python 3.10+ (found $("$PY" --version 2>&1))." >&2; exit 1; }

echo "==> creating $VIOLA_HOME/venv"
"$PY" -m venv "$VIOLA_HOME/venv"
VPY="$VIOLA_HOME/venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

if [ "$(uname -s)" != "Darwin" ]; then
  echo "==> installing CPU-only PyTorch"
  "$VPY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> installing viola"
"$VPY" -m pip install --quiet "viola[model,bedrock] @ $(case "$VIOLA_SRC" in git+*|http*) echo "$VIOLA_SRC";; *) echo "file://$VIOLA_SRC";; esac)"

mkdir -p "$BIN_DIR"
ln -sf "$VIOLA_HOME/venv/bin/viola" "$BIN_DIR/viola"
echo "==> linked $BIN_DIR/viola"

"$BIN_DIR/viola" setup

case ":$PATH:" in *":$BIN_DIR:"*) ;; *) echo "Add $BIN_DIR to your PATH to use \`viola\`." ;; esac
cat <<EOF

Viola is installed. Next:
  viola install claude-code     # and/or: viola install codex
  viola start                   # or: viola autostart (start at login)
  viola status
EOF
