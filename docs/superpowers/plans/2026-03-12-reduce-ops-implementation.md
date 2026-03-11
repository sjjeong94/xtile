# XT Reduce Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xt.reduce_sum` and `xt.reduce_max` for last-dimension reductions.

**Architecture:** Model reductions as separate unary xt ops whose result keeps the same rank as the input but forces the last dimension to `1`. Verify only static ranked tensors, then lower by iterating the result prefix and reducing across the input's final dimension.

**Tech Stack:** MLIR ODS/C++, lit/FileCheck tests, xt lower-to-loops pass

---

## Chunk 1: Test First

### Task 1: Add failing reduce tests

**Files:**
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing test**
Add parse and lowering coverage for `xt.reduce_sum` and `xt.reduce_max`, plus an invalid case for a wrong reduction result shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target check-xt`
Expected: FAIL because the new reduce ops are undefined.

## Chunk 2: Implementation

### Task 2: Add op surface and verifier

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`

- [ ] **Step 1: Write minimal implementation**
Add `xt.reduce_sum` and `xt.reduce_max` as unary ops with parser/printer and verifier checks that only the last dimension is reduced to `1`.

- [ ] **Step 2: Run focused tests**

Run: `cmake --build build --target check-xt`
Expected: parser/verifier improve while lowering still fails.

### Task 3: Add lowering

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`

- [ ] **Step 1: Write minimal implementation**
Lower reductions by iterating output indices and reducing over the input tensor's final dimension.

- [ ] **Step 2: Run verification**

Run: `cmake --build build --target check-xt`
Expected: PASS
