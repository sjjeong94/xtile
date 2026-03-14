#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_ROOT="${XTILE_LLVM_ROOT:-$REPO_ROOT/../llvm-project}"
LLVM_REF="${LLVM_REF:-llvmorg-20.1.8}"
LLVM_BUILD_DIR="${XTILE_LLVM_BUILD_DIR:-$LLVM_ROOT/build}"
LLVM_INSTALL_DIR="${XTILE_LLVM_INSTALL_DIR:-$LLVM_ROOT/install}"
LLVM_SOURCE_DIR="$LLVM_ROOT/llvm"

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 1
  fi
}

require_git_repo() {
  local path="$1"
  if [[ ! -d "$path/.git" ]]; then
    echo "existing path is not a git repository: $path" >&2
    exit 1
  fi
}

require_tool git
require_tool cmake

if command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR_ARGS=(-G Ninja)
else
  CMAKE_GENERATOR_ARGS=()
fi

if [[ ! -e "$LLVM_ROOT" ]]; then
  git clone https://github.com/llvm/llvm-project.git "$LLVM_ROOT"
elif [[ -d "$LLVM_ROOT" ]]; then
  require_git_repo "$LLVM_ROOT"
else
  echo "llvm root exists but is not a directory: $LLVM_ROOT" >&2
  exit 1
fi

git -C "$LLVM_ROOT" fetch --tags origin "$LLVM_REF"
git -C "$LLVM_ROOT" checkout "$LLVM_REF"

mkdir -p "$LLVM_BUILD_DIR" "$LLVM_INSTALL_DIR"

cmake "${CMAKE_GENERATOR_ARGS[@]}" \
  -S "$LLVM_SOURCE_DIR" \
  -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR"

cmake --build "$LLVM_BUILD_DIR" --target install
