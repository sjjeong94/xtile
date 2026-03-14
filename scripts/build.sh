#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MLIR_DIR="$(cd "$REPO_ROOT/../llvm-project/build/lib/cmake/mlir" 2>/dev/null && pwd || true)"
DEFAULT_MLIR_PYTHON_ROOT="$REPO_ROOT/../llvm-project/build/tools/mlir/python_packages/mlir_core"
BUILD_DIR="${XTILE_BUILD_DIR:-$REPO_ROOT/build}"
MLIR_DIR="${MLIR_DIR:-$DEFAULT_MLIR_DIR}"
MLIR_PYTHON_ROOT="${XTILE_MLIR_PYTHON_ROOT:-$DEFAULT_MLIR_PYTHON_ROOT}"
LLVM_LIT_BIN="${XTILE_LLVM_LIT_BIN:-$REPO_ROOT/tools/llvm-lit-wrapper}"

require_dir() {
  local path="$1"
  if [[ -z "$path" || ! -d "$path" ]]; then
    echo "missing required directory: $path" >&2
    exit 1
  fi
}

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 1
  fi
}

require_dir "$MLIR_DIR"
require_dir "$MLIR_PYTHON_ROOT"
require_tool python3
require_tool "$LLVM_LIT_BIN"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DMLIR_DIR="$MLIR_DIR"
cmake --build "$BUILD_DIR"
"$LLVM_LIT_BIN" -sv -j 1 "$BUILD_DIR/test"
python3 -c "import pytest" >/dev/null 2>&1 || {
  echo "missing required Python module: pytest" >&2
  exit 1
}
PYTHONPATH="$REPO_ROOT/python:$MLIR_PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest python/tests -v
