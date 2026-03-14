# Nova Matmul Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `nova.matmul` and convert `xt.matmul` into it.

**Architecture:** Extend the existing nova op set with one dedicated matmul op. Reuse the current split between the C++ `xt-opt` pass and the Python string-based `xt.to_nova(...)` helper so both CLI and Python flows expose the same Nova IR surface. Keep scale and bias explicit as `f32` SSA constants emitted at the rewrite site.

**Tech Stack:** MLIR ODS/TableGen, C++ rewrite patterns, Python regex/string rewriting, `lit`, `pytest`

---

## Chunk 1: Tests First

### Task 1: Add failing regression coverage

**Files:**
- Modify: `test/xt/to_nova.mlir`
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Add lit checks for `xt.matmul` lowering to `nova.matmul`**
- [ ] **Step 2: Add Python checks that `xt.to_nova(...)` emits `nova.matmul`**
- [ ] **Step 3: Run focused tests to verify they fail before implementation**

## Chunk 2: Implementation

### Task 2: Add the op and conversion logic

**Files:**
- Modify: `include/nova/NovaOps.td`
- Modify: `lib/xt/XTToNova.cpp`
- Modify: `python/xtile/nova.py`

- [ ] **Step 1: Define `nova.matmul` in ODS**
- [ ] **Step 2: Extend the C++ pass with `xt.matmul` -> `nova.matmul` rewrite**
- [ ] **Step 3: Extend the Python conversion helper with `xt.matmul` rewrite**
- [ ] **Step 4: Re-run focused verification**
