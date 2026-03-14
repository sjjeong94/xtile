# Build Script Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `scripts/build.sh` helper that configures and builds the C++ project with a repository-relative default `MLIR_DIR`.

**Architecture:** Keep the script narrowly scoped: path resolution, validation, configure, build. Reuse the existing shell style from `scripts/build_wheel.sh` so the repo has one consistent convention for helper scripts.

**Tech Stack:** Bash, CMake

---

## Chunk 1: Script Creation

### Task 1: Add the failing check and implement the script

**Files:**
- Create: `scripts/build.sh`
- Reference: `scripts/build_wheel.sh`
- Reference: `CMakeLists.txt`

- [ ] **Step 1: Write the failing check**

Run: `bash scripts/build.sh`
Expected: FAIL because the script does not exist yet.

- [ ] **Step 2: Verify the failing check**

Run: `bash scripts/build.sh`
Expected: shell reports `No such file or directory`.

- [ ] **Step 3: Write the minimal implementation**

Create a Bash script that:
- resolves `REPO_ROOT`
- sets `BUILD_DIR="${XTILE_BUILD_DIR:-$REPO_ROOT/build}"`
- sets `MLIR_DIR="${MLIR_DIR:-$REPO_ROOT/../llvm-project/build/lib/cmake/mlir}"`
- validates the chosen `MLIR_DIR`
- runs `cmake -S` then `cmake --build`

- [ ] **Step 4: Run verification**

Run: `bash -n scripts/build.sh`
Expected: PASS

Run: `bash scripts/build.sh`
Expected: either config/build proceeds or the script fails with the explicit missing `MLIR_DIR` message.
