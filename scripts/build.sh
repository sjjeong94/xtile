#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MLIR_DIR="$(cd "$REPO_ROOT/../llvm-project/build/lib/cmake/mlir" 2>/dev/null && pwd || true)"
BUILD_DIR="${XTILE_BUILD_DIR:-$REPO_ROOT/build}"
MLIR_DIR="${MLIR_DIR:-$DEFAULT_MLIR_DIR}"

require_dir() {
  local path="$1"
  if [[ -z "$path" || ! -d "$path" ]]; then
    echo "missing required directory: $path" >&2
    exit 1
  fi
}

require_dir "$MLIR_DIR"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DMLIR_DIR="$MLIR_DIR"
cmake --build "$BUILD_DIR"
