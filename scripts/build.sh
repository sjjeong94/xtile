#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MLIR_DIR="$(cd "$REPO_ROOT/../llvm-project/build/lib/cmake/mlir" 2>/dev/null && pwd || true)"
DEFAULT_LLVM_LIT_BIN="$REPO_ROOT/../llvm-project/build/bin/llvm-lit"
BUILD_DIR="${XTILE_BUILD_DIR:-$REPO_ROOT/build}"
MLIR_DIR="${MLIR_DIR:-$DEFAULT_MLIR_DIR}"
LLVM_LIT_BIN="${XTILE_LLVM_LIT_BIN:-}"

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

resolve_llvm_lit() {
  if [[ -n "$LLVM_LIT_BIN" ]]; then
    require_tool "$LLVM_LIT_BIN"
    printf '%s\n' "$LLVM_LIT_BIN"
    return
  fi

  if [[ -x "$DEFAULT_LLVM_LIT_BIN" ]]; then
    printf '%s\n' "$DEFAULT_LLVM_LIT_BIN"
    return
  fi

  require_tool llvm-lit
  command -v llvm-lit
}

require_dir "$MLIR_DIR"
LLVM_LIT_BIN="$(resolve_llvm_lit)"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DMLIR_DIR="$MLIR_DIR"
cmake --build "$BUILD_DIR"
"$LLVM_LIT_BIN" -sv -j 1 "$BUILD_DIR/test"
