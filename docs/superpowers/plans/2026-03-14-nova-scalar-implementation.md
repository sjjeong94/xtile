# Nova Scalar Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `nova.scalar` op and lower `xt` binary ops with compile-time constant RHS values to it.

**Architecture:** Define `nova.scalar` as a dedicated unary Nova op with `mode` and `rhs` attributes. Update the C++ pass and Python rewrite to detect compile-time RHS constants first, then fall back to existing `nova.elementwise`/`nova.broadcast` behavior.

**Tech Stack:** MLIR TableGen/C++, lit/FileCheck, Python MLIR text rewrite, pytest

---

## Chunk 1: Tests First

### Task 1: Add MLIR regression coverage

**Files:**
- Modify: `test/xt/to_nova.mlir`

- [ ] **Step 1: Write the failing test**
  Add one function where the RHS is an `arith.constant` splat tensor and the expected output is `nova.scalar`.

- [ ] **Step 2: Run test to verify it fails**
  Run: `bash -lc './build/bin/xt-opt --xt-to-nova test/xt/to_nova.mlir | /home/sjjeong94/projects/llvm-project/build/bin/FileCheck test/xt/to_nova.mlir'`
  Expected: FAIL because `nova.scalar` is not emitted yet.

### Task 2: Add Python regression coverage

**Files:**
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**
  Add a Python-side `xt.to_nova()` test that includes a compile-time constant RHS and expects `nova.scalar`.

- [ ] **Step 2: Run test to verify it fails**
  Run: `python3 -m pytest python/tests/test_python_dsl.py -k nova_scalar -v`
  Expected: FAIL because Python rewrite still leaves `xt.*`.

## Chunk 2: Dialect and Lowering

### Task 3: Define the op

**Files:**
- Modify: `include/nova/NovaOps.td`

- [ ] **Step 1: Add `Nova_ScalarOp`**
  Define a pure op with one ranked tensor input, `I32Attr:$mode`, `F32Attr:$rhs`, and one ranked tensor result.

### Task 4: Lower in C++

**Files:**
- Modify: `lib/xt/XTToNova.cpp`

- [ ] **Step 1: Implement constant extraction helper**
  Detect `arith.constant` RHS values and extract an `f32` scalar from scalar or splat tensor attributes.

- [ ] **Step 2: Lower matching ops to `nova.scalar`**
  Rewrite `xt.add`, `xt.mul`, `xt.sub` to `nova.scalar` before the existing elementwise/broadcast logic.

- [ ] **Step 3: Preserve existing behavior**
  Leave non-constant scalar-like cases unchanged and keep current tensor/tensor lowering intact.

### Task 5: Mirror the behavior in Python

**Files:**
- Modify: `python/xtile/nova.py`

- [ ] **Step 1: Extend rewrite logic**
  Detect constant RHS values in the dumped MLIR text and emit `nova.scalar` with the same mode mapping.

- [ ] **Step 2: Keep fallback behavior unchanged**
  Non-constant scalar-like operands should still remain as `xt.*`.

## Chunk 3: Verification

### Task 6: Focused verification

**Files:**
- Test: `test/xt/to_nova.mlir`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Run MLIR regression**
  Run: `bash -lc './build/bin/xt-opt --xt-to-nova test/xt/to_nova.mlir | /home/sjjeong94/projects/llvm-project/build/bin/FileCheck test/xt/to_nova.mlir'`
  Expected: PASS

- [ ] **Step 2: Run Python regression**
  Run: `python3 -m pytest python/tests/test_python_dsl.py -k 'nova and scalar' -v`
  Expected: PASS

### Task 7: Full verification

**Files:**
- Test: `./scripts/build.sh`

- [ ] **Step 1: Run full build and test flow**
  Run: `./scripts/build.sh`
  Expected: lit suite passes and pytest suite passes with 0 failures.
