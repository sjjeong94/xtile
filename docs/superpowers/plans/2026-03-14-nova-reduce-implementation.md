# Nova Reduce Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `nova.reduce` and convert `xt.reduce_sum` / `xt.reduce_max` into it.

**Architecture:** Extend the existing nova op set with one unary op and reuse the current xt-to-nova split: a C++ pass for `xt-opt` and a Python string-based post-pass for the Python CLI path. Keep the mode encoding small and consistent with existing nova op attributes.

**Tech Stack:** MLIR ODS/TableGen, C++ rewrite patterns, Python regex/string rewriting, `lit`, `pytest`

---

## Chunk 1: Tests First

### Task 1: Add failing lit and Python expectations

**Files:**
- Modify: `test/xt/to_nova.mlir`
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Add lit checks for `nova.reduce`**
- [ ] **Step 2: Add Python checks for softmax `nova.reduce` output**
- [ ] **Step 3: Add CLI check that `--xt-to-nova` prints `nova.reduce`**
- [ ] **Step 4: Run focused tests**

Run: `cmake --build build --target check-xt`
Expected: FAIL until conversion support is implemented.

Run: `python3 -m pytest python/tests/test_python_dsl.py -k nova -v`
Expected: FAIL until conversion support is implemented.

## Chunk 2: Implementation

### Task 2: Add the op and conversion logic

**Files:**
- Modify: `include/nova/NovaOps.td`
- Modify: `lib/xt/XTToNova.cpp`
- Modify: `python/xtile/nova.py`

- [ ] **Step 1: Define `nova.reduce` in ODS**
- [ ] **Step 2: Extend the C++ pass with `xt.reduce_sum` / `xt.reduce_max` rewrites**
- [ ] **Step 3: Extend the Python conversion helper with reduce rewrites**
- [ ] **Step 4: Re-run verification**
