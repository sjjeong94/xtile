# XT Reshape And Transpose Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xt.reshape` and `xt.transpose` ops that cover the documented reshape/transpose tile example.

**Architecture:** Model both as pure tensor ops. `xt.reshape` verifies ranked static tensors preserve element count and lowers by remapping linearized indices. `xt.transpose` verifies rank-3 tensors preserve element type and swaps the last two dimensions, then lowers by reading the source tensor with permuted indices.

**Tech Stack:** MLIR ODS/C++, xt lower-to-loops pass, lit/FileCheck tests

---

## Chunk 1: Tests First

### Task 1: Add failing op coverage

**Files:**
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing test**
Add parse and lowering coverage for the documented reshape/transpose flow plus invalid cases for mismatched reshape element count and invalid transpose result shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target check-xt`
Expected: FAIL because `xt.reshape` and `xt.transpose` are undefined.

## Chunk 2: Op Surface And Verification

### Task 2: Add op definitions and verifiers

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`

- [ ] **Step 1: Write minimal implementation**
Add pure unary ops with parser/printer support. Verify reshape preserves ranked static element count and element type. Verify transpose accepts rank-3 static tensors and swaps dimensions 1 and 2 in the result while preserving dim 0 and element type.

- [ ] **Step 2: Run tests**

Run: `cmake --build build --target check-xt`
Expected: parse/verifier coverage improves while lowering still fails.

## Chunk 3: Lowering

### Task 3: Lower reshape and transpose

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`

- [ ] **Step 1: Write minimal implementation**
Lower reshape by converting result indices to a linear offset and reconstructing source indices. Lower transpose by extracting from the source tensor with indices `[i, k, j]` for result indices `[i, j, k]`.

- [ ] **Step 2: Run verification**

Run: `cmake --build build --target check-xt`
Expected: PASS
