# Build Script Test Run Design

**Goal:** Extend `scripts/build.sh` so it not only configures and builds the C++ project, but also runs the repository's C++ and Python regression tests.

## Scope

- Keep the existing configure/build flow
- Add MLIR lit regression execution
- Add Python `pytest` execution with the correct MLIR Python path

## Behavior

- Resolve the same repository-relative `MLIR_DIR` as today
- After a successful build, run:
  - `cmake --build "$BUILD_DIR" --target check-xt`
  - `python3 -m pytest python/tests -v`
- Before Python tests:
  - verify the MLIR Python package root exists at
    `../llvm-project/build/tools/mlir/python_packages/mlir_core`
  - export `PYTHONPATH` including:
    - `<repo>/python`
    - MLIR Python root
- Fail early with clear messages when Python test prerequisites are missing

## Validation

- `bash -n scripts/build.sh` must pass
- Running `bash scripts/build.sh` must now execute build, `check-xt`, and Python tests in sequence
