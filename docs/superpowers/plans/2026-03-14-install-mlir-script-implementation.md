# Install MLIR Script Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/install_mlir.sh` safe to rerun against the repository-relative `../llvm-project` tree.

**Architecture:** Convert the current ad-hoc sequence into a guarded Bash script with path resolution, repo validation, conditional clone/fetch, and repeatable CMake configure/build/install steps. Keep defaults aligned with the rest of this repository so `scripts/build.sh` can continue assuming `../llvm-project/build/lib/cmake/mlir`.

**Tech Stack:** Bash, git, CMake

---

## Chunk 1: Script Rewrite

### Task 1: Replace the one-shot script with a reusable version

**Files:**
- Modify: `scripts/install_mlir.sh`

- [ ] **Step 1: Write the failing check**

Run: `bash -n scripts/install_mlir.sh`
Expected: PASS syntactically, but script structure still shows one-shot behavior like unconditional clone and `apt`.

- [ ] **Step 2: Verify current behavior is not reusable**

Inspect the script and confirm it:
- runs `sudo apt`
- unconditionally clones
- uses `cd ..` statefully

- [ ] **Step 3: Write the minimal reusable implementation**

Rewrite the script to:
- resolve paths from the repo root
- conditionally clone or reuse `../llvm-project`
- validate repo state
- fetch/checkout `LLVM_REF`
- rerun configure and install safely

- [ ] **Step 4: Run verification**

Run: `bash -n scripts/install_mlir.sh`
Expected: PASS

Run: `bash scripts/install_mlir.sh`
Expected: reuses existing `../llvm-project` if present, then configures/builds/installs.
