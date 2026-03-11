# XT Conv2D Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xt.conv2d` to the xt dialect with verifier, assembly support, loop lowering, and regression tests matching the documented example.

**Architecture:** Extend the existing xt op surface in the same style as `xt.matmul` and other tensor ops. Verify a constrained 4D NHWC x HWIO convolution shape and lower it directly to nested loops with explicit padding, stride, and dilation index math.

**Tech Stack:** MLIR ODS/C++, lit/FileCheck tests, xt lower-to-loops pass

---

## Chunk 1: Tests First

### Task 1: Add failing conv2d parser and lowering tests

**Files:**
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing test**
Add `xt.conv2d` coverage to parse, lower, and invalid tests with the documented NHWC/HWIO example and verifier failures for malformed attributes or shapes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target check-xt`
Expected: FAIL because `xt.conv2d` is unknown or unsupported.

### Task 2: Implement op surface and verifier

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`

- [ ] **Step 1: Write minimal implementation**
Add the op definition, parser/printer helpers for `pad/stride/dilation`, and verifier checks for rank-4 static tensors, attribute lengths, padding/stride/dilation validity, channel compatibility, and result shape.

- [ ] **Step 2: Run focused tests**

Run: `cmake --build build --target check-xt`
Expected: parse/invalid coverage passes while lowering still fails.

## Chunk 2: Lowering

### Task 3: Lower conv2d to loops

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`
- Test: `test/xt/lower.mlir`

- [ ] **Step 1: Implement lowering**
Lower `xt.conv2d` by iterating output `N/H/W/Cout` and accumulating over `Kh/Kw/Cin`, applying zero-padding and `stride`/`dilation` index math.

- [ ] **Step 2: Run focused tests**

Run: `cmake --build build --target check-xt`
Expected: PASS
