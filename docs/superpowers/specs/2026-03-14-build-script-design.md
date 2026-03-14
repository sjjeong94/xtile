# Build Script Design

**Goal:** Add a repository-local `scripts/build.sh` that configures and builds the C++ MLIR project with a default `MLIR_DIR` resolved from `../llvm-project/...`.

## Scope

- Add `scripts/build.sh`
- Keep `scripts/build_wheel.sh` unchanged
- Default build directory to `<repo>/build`
- Default `MLIR_DIR` to a repository-relative path under `../llvm-project`
- Allow callers to override `MLIR_DIR` and build directory through environment variables

## Behavior

- Resolve `REPO_ROOT` from the script location
- Resolve `DEFAULT_MLIR_DIR` from `"$REPO_ROOT/../llvm-project/build/lib/cmake/mlir"`
- Use `XTILE_BUILD_DIR` when provided, otherwise `"$REPO_ROOT/build"`
- Use `MLIR_DIR` when provided, otherwise the resolved default path
- Fail early with a clear error if the chosen `MLIR_DIR` does not exist
- Run:
  - `cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DMLIR_DIR="$MLIR_DIR"`
  - `cmake --build "$BUILD_DIR"`

## Validation

- `bash -n scripts/build.sh` must succeed
- Running the script in an environment without the relative MLIR tree should fail with the explicit missing-path error
