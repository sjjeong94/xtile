#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MLIR_DIR="$(cd "$REPO_ROOT/../llvm-project/build/lib/cmake/mlir" 2>/dev/null && pwd || true)"
BUILD_DIR="${XTILE_BUILD_DIR:-$REPO_ROOT/build}"
DIST_DIR="${XTILE_WHEEL_DIST_DIR:-$REPO_ROOT/dist}"
MLIR_DIR="${MLIR_DIR:-$DEFAULT_MLIR_DIR}"
PYTHON_BIN="${XTILE_PYTHON_BIN:-python3}"
WHEEL_VERSION="${XTILE_WHEEL_VERSION:-0.1.0}"
CLEAN_BUILD=0

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

require_python_module() {
  local module="$1"
  if ! "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$module") else 1)
PY
  then
    echo "missing required python module: $module" >&2
    exit 1
  fi
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --build-dir PATH   CMake build directory (default: $BUILD_DIR)
  --dist-dir PATH    Wheel output directory (default: $DIST_DIR)
  --mlir-dir PATH    MLIRConfig.cmake directory (default: $MLIR_DIR)
  --python PATH      Python executable (default: $PYTHON_BIN)
  --version VERSION  Wheel version (default: $WHEEL_VERSION)
  --clean            Remove the build directory before configuring
  -h, --help         Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --dist-dir)
      DIST_DIR="$2"
      shift 2
      ;;
    --mlir-dir)
      MLIR_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --version)
      WHEEL_VERSION="$2"
      shift 2
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_tool cmake
require_tool "$PYTHON_BIN"
require_dir "$MLIR_DIR"
require_python_module nanobind
require_python_module wheel

PYTHON_EXE="$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.executable)
PY
)"

PYTHON_BIN_DIR="$(dirname "$PYTHON_EXE")"
export PATH="$PYTHON_BIN_DIR:$PATH"

PYTHON_PLATFORM="$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.platform)
PY
)"

if [[ "$CLEAN_BUILD" -eq 1 ]]; then
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$DIST_DIR"

if [[ "$PYTHON_PLATFORM" == "linux" ]]; then
  require_python_module auditwheel
  require_tool patchelf
fi

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DXT_ENABLE_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$PYTHON_EXE"
cmake --build "$BUILD_DIR" --target xt

EXTENSION_PATH="$("$PYTHON_BIN" - <<PY
from pathlib import Path

build_python_dir = Path(r"$BUILD_DIR") / "python"
candidates = sorted(build_python_dir.glob("xtile*.so")) + sorted(build_python_dir.glob("xtile*.pyd"))
if not candidates:
    raise SystemExit("missing built xtile extension in " + str(build_python_dir))
print(candidates[0])
PY
)"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

WHEEL_PATH="$("$PYTHON_BIN" - <<PY
import base64
import csv
import hashlib
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import zipfile

repo_root = Path(r"$REPO_ROOT")
dist_dir = Path(r"$DIST_DIR")
extension_path = Path(r"$EXTENSION_PATH")
version = "$WHEEL_VERSION"
name = "xtile"

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
soabi = sysconfig.get_config_var("SOABI") or ""
if soabi.startswith("cpython-"):
    abi_suffix = soabi.split("-", 2)[1]
    abi_tag = f"cp{abi_suffix}"
else:
    abi_tag = py_tag
platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")

wheel_basename = f"{name}-{version}-{py_tag}-{abi_tag}-{platform_tag}"
stage_dir = Path(r"$TMPDIR") / wheel_basename
stage_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(extension_path, stage_dir / extension_path.name)

dist_info = stage_dir / f"{name}-{version}.dist-info"
dist_info.mkdir()
(dist_info / "METADATA").write_text(
    "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            "Summary: xtile MLIR Python bindings",
            "",
        ]
    ),
    encoding="utf-8",
)
(dist_info / "WHEEL").write_text(
    "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: scripts/build_wheel.sh",
            "Root-Is-Purelib: false",
            f"Tag: {py_tag}-{abi_tag}-{platform_tag}",
            "",
        ]
    ),
    encoding="utf-8",
)

records = []
for path in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
    relpath = path.relative_to(stage_dir).as_posix()
    digest = hashlib.sha256(path.read_bytes()).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    records.append((relpath, f"sha256={b64}", str(path.stat().st_size)))

record_path = dist_info / "RECORD"
records.append((record_path.relative_to(stage_dir).as_posix(), "", ""))
with record_path.open("w", encoding="utf-8", newline="") as handle:
    csv.writer(handle).writerows(records)

wheel_path = dist_dir / f"{wheel_basename}.whl"
with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
        zf.write(path, path.relative_to(stage_dir).as_posix())

print(wheel_path)
PY
)"

FINAL_WHEEL_PATH="$WHEEL_PATH"

if [[ "$PYTHON_PLATFORM" == "linux" ]]; then
  REPAIRED_DIR="$DIST_DIR/repaired"
  mkdir -p "$REPAIRED_DIR"
  "$PYTHON_BIN" -m auditwheel repair --wheel-dir "$REPAIRED_DIR" "$WHEEL_PATH"
  REPAIRED_WHEEL="$("$PYTHON_BIN" - <<PY
from pathlib import Path
repaired_dir = Path(r"$REPAIRED_DIR")
candidates = sorted(repaired_dir.glob("xtile-*.whl"))
if not candidates:
    raise SystemExit("auditwheel did not produce a repaired xtile wheel")
print(candidates[-1])
PY
)"
  FINAL_WHEEL_PATH="$REPAIRED_WHEEL"
else
  echo "built an unrepaired non-Linux wheel" >&2
fi

echo "built wheel: $FINAL_WHEEL_PATH"
