# Python DSL Float Scalar Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support Python float scalar bindings in xt kernel conversion so the updated `layernorm.py` converts cleanly.

**Architecture:** Keep the public DSL unchanged and implement the feature by promoting scalar floats to dense tensor constants at parse time. This avoids introducing a separate scalar IR layer and lets the existing `FullOp` builder path do the emission work.

**Tech Stack:** Python AST parsing, dataclasses, existing xt MLIR builder, `pytest`

---

## Chunk 1: Tests First

### Task 1: Strengthen the layernorm regression

**Files:**
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing expectation**

Extend the `layernorm_kernel` test to assert the dumped MLIR contains the float scalar value in a dense constant, e.g. `0.062500`.

- [ ] **Step 2: Run the focused test**

Run: `python3 -m pytest python/tests/test_python_dsl.py::test_convert_layernorm_kernel -v`
Expected: FAIL until float scalar bindings are supported.

## Chunk 2: Implementation

### Task 2: Add scalar float parsing and promotion

**Files:**
- Modify: `python/xtile/ast_parser.py`

- [ ] **Step 1: Add a scalar float environment value type**
- [ ] **Step 2: Accept float constant assignments**
- [ ] **Step 3: Promote scalar float operands to `FullOp` when used in binary ops**
- [ ] **Step 4: Re-run verification**
