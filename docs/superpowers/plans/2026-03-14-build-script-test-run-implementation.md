# Build Script Test Run Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/build.sh` perform the repository's standard verification steps after building.

**Architecture:** Keep the script linear and explicit: resolve paths, configure, build, run `check-xt`, then run Python tests with a prepared `PYTHONPATH`. Reuse the repository-relative llvm-project assumptions already established in the existing helper scripts.

**Tech Stack:** Bash, CMake, pytest

---

## Chunk 1: Script Extension

### Task 1: Add test execution to `build.sh`

**Files:**
- Modify: `scripts/build.sh`

- [ ] **Step 1: Establish the failing baseline**

Inspect the script and confirm it currently stops after `cmake --build`.

- [ ] **Step 2: Write the minimal implementation**

Add:
- MLIR Python root resolution
- `python3` / `pytest` tool checks
- `check-xt` invocation
- `pytest python/tests -v` invocation with the required `PYTHONPATH`

- [ ] **Step 3: Run verification**

Run: `bash -n scripts/build.sh`
Expected: PASS

Run: `bash scripts/build.sh`
Expected: build, `check-xt`, and Python tests run in sequence.
