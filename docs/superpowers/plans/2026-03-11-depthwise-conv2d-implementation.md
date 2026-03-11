# XT Depthwise Conv2D Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xt.depthwise_conv2d` with verifier, lowering, and regression tests for the documented depth_multiplier=1 NHWC example.

**Architecture:** Model depthwise convolution as its own xt op with the same `pad/stride/dilation` surface as `xt.conv2d`. Reuse the existing convolution-style attribute handling and output-size calculation, but verify and lower the depthwise-specific channel semantics separately.

**Tech Stack:** MLIR ODS/C++, lit/FileCheck tests, xt lower-to-loops pass

---

## Chunk 1: Test First

### Task 1: Add failing depthwise conv tests

**Files:**
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing test**
Add one parse case, one lowering case, and one invalid verifier case for `xt.depthwise_conv2d`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target check-xt`
Expected: FAIL because `xt.depthwise_conv2d` is not defined yet.

## Chunk 2: Implementation

### Task 2: Add op surface and verifier

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`

- [ ] **Step 1: Write minimal implementation**
Add the op, parser/printer, and verifier for rank-4 static tensors with `KhxKwx1xC` filter shape and identical input/output channels.

- [ ] **Step 2: Run focused tests**

Run: `cmake --build build --target check-xt`
Expected: parser/verifier progress, lowering still missing.

### Task 3: Add lowering

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`

- [ ] **Step 1: Write minimal implementation**
Lower output `N/H/W/C` directly with `Kh/Kw` reduction and per-channel indexing.

- [ ] **Step 2: Run verification**

Run: `cmake --build build --target check-xt`
Expected: PASS
