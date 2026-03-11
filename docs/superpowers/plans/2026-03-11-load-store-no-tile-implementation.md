# XT Load/Store No-Tile Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `tile` attribute from `xt.load` and `xt.store`, deriving tile extents from tensor types instead.

**Architecture:** Simplify the IR surface so tensor shapes are the single source of truth for tile extents. Keep `shared` on `xt.load`, update parser/verifier/lowering accordingly, and convert docs and lit tests to the new syntax.

**Tech Stack:** MLIR ODS/C++, lit/FileCheck tests, xt lower-to-loops pass

---

## Chunk 1: Test-First Surface Change

### Task 1: Remove `tile` from docs and tests

**Files:**
- Modify: `xtile.md`
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing test**
Remove `tile = [...]` from all `xt.load` and `xt.store` uses in docs/tests, and delete invalid cases that only checked tile-shape mismatches.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target check-xt`
Expected: FAIL because the parser/verifier still require the `tile` attribute.

## Chunk 2: Implementation

### Task 2: Remove tile from op definitions and verifier

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`

- [ ] **Step 1: Write minimal implementation**
Drop the `tile` attribute from `xt.load`/`xt.store`, keep optional `shared` for `xt.load`, and validate using tensor shape/rank directly.

- [ ] **Step 2: Run focused tests**

Run: `cmake --build build --target check-xt`
Expected: parse/invalid move forward while lowering still needs updates.

### Task 3: Update lowering

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`

- [ ] **Step 1: Write minimal implementation**
Compute base offsets from the load/store tensor shape instead of a `tile` attribute.

- [ ] **Step 2: Run verification**

Run: `cmake --build build --target check-xt`
Expected: PASS
