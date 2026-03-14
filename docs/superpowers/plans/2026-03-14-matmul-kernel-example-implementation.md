# Matmul Kernel Example Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable matmul example kernel under `python/kernels/` and verify it converts through the existing DSL.

**Architecture:** Reuse the already-supported row/column tiled matmul pattern from the DSL tests. No parser or IR changes are needed; the work is limited to adding the example module, exporting it, and asserting the example converts to MLIR.

**Tech Stack:** Python stdlib, existing `xtile` DSL, `pytest`

---

## Chunk 1: Tests First

### Task 1: Add a failing example-kernel conversion test

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

Add a test that imports `kernels.matmul.matmul_kernel`, converts it, and checks for:
- `func.func @matmul_kernel`
- two `xt.load` ops
- `xt.matmul`
- `xt.store`
- `tensor<64x64xf32>`

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py -k 'matmul_kernel_example' -q`
Expected: FAIL because `python/kernels/matmul.py` does not exist yet

## Chunk 2: Minimal Implementation

### Task 2: Add the example module and export

**Files:**
- Create: `python/kernels/matmul.py`
- Modify: `python/kernels/__init__.py`

- [ ] **Step 3: Write minimal implementation**

Add the existing supported matmul example as a kernel module under `python/kernels/` and export `matmul_kernel` from `python/kernels/__init__.py`.

- [ ] **Step 4: Run targeted test to verify it passes**

Run: `pytest python/tests/test_python_dsl.py -k 'matmul_kernel_example' -q`
Expected: PASS

## Chunk 3: Full Verification

### Task 3: Run regression checks

**Files:**
- No additional file changes expected

- [ ] **Step 5: Run DSL regression suite**

Run: `pytest python/tests/test_python_dsl.py -q`
Expected: PASS

- [ ] **Step 6: Run wheel regression suite**

Run: `pytest python/tests/test_wheel_build.py -q`
Expected: PASS
