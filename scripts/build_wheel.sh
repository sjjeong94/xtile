#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MLIR_ROOT="/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core"
MLIR_ROOT="${1:-${XTILE_MLIR_PYTHON_ROOT:-$DEFAULT_MLIR_ROOT}}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  DEFAULT_PYTHON_BIN="$(command -v python || command -v python3)"
fi
PYTHON_BIN="${XTILE_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
DIST_DIR="${XTILE_DIST_DIR:-$REPO_ROOT/dist}"
BUILD_ROOT="${XTILE_BUILD_ROOT:-$REPO_ROOT/.wheel-build}"
STAGING_DIR="$BUILD_ROOT/staging"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
}

require_path "$PYTHON_BIN"
require_path "$REPO_ROOT/python/xtile"
require_path "$REPO_ROOT/pyproject.toml"
require_path "$REPO_ROOT/MANIFEST.in"
require_path "$REPO_ROOT/setup.py"
require_path "$MLIR_ROOT/mlir/ir.py"
require_path "$MLIR_ROOT/mlir/_mlir_libs"

"$PYTHON_BIN" -m build --version >/dev/null

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR" "$DIST_DIR"

cp "$REPO_ROOT/pyproject.toml" "$STAGING_DIR/pyproject.toml"
cp "$REPO_ROOT/MANIFEST.in" "$STAGING_DIR/MANIFEST.in"
cp "$REPO_ROOT/setup.py" "$STAGING_DIR/setup.py"
cp -R "$REPO_ROOT/python/xtile" "$STAGING_DIR/xtile"
cp -R "$MLIR_ROOT/mlir" "$STAGING_DIR/mlir"

if [[ ! -f "$STAGING_DIR/mlir/__init__.py" ]]; then
  : > "$STAGING_DIR/mlir/__init__.py"
fi

rm -rf "$STAGING_DIR"/xtile/__pycache__
find "$STAGING_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGING_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

"$PYTHON_BIN" -m build --no-isolation --wheel --outdir "$DIST_DIR" "$STAGING_DIR"
