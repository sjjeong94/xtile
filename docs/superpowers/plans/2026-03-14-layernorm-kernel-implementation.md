# LayerNorm Kernel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a row-wise `layernorm` example kernel to `python/kernels/` and extend the Python DSL only as much as needed to express it.

**Architecture:** Keep the kernel shape consistent with the current examples: load a `16x16` tile, reduce over the last dimension, and rely on existing broadcast rules for row-wise normalization. Add one minimal DSL primitive for constant tensor creation so the kernel can express multiplication by `1/16` without introducing unrelated semantics.

**Tech Stack:** Python stdlib, existing `xtile` DSL/parser/IR builder, `pytest`

---

## Chunk 1: Red Tests

### Task 1: Add failing tests for constant tensors and layernorm conversion

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

Add:
- a minimal kernel that uses a constant tensor scale and asserts conversion contains an `arith.constant` tensor
- a `layernorm_kernel` conversion test that checks for `reduce_sum`, `sub`, `mul`, `rsqrt`, and the expected tensor shapes

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py -k 'constant_tensor or layernorm' -q`
Expected: FAIL because the DSL cannot yet express constant tensors and the kernel module does not exist

## Chunk 2: Minimal Implementation

### Task 2: Implement constant tensor support

**Files:**
- Modify: `python/xtile/dsl.py`
- Modify: `python/xtile/__init__.py`
- Modify: `python/xtile/ast_parser.py`
- Modify: `python/xtile/ir_builder.py`

- [ ] **Step 3: Write minimal implementation**

Add an `xt.full(shape=(...), value=...)` DSL function that:
- accepts a float constant
- returns a tensor with the given shape and `f32` element type
- lowers to an `arith.constant` ranked tensor value

- [ ] **Step 4: Run targeted tests to verify it passes**

Run: `pytest python/tests/test_python_dsl.py -k 'constant_tensor' -q`
Expected: PASS

### Task 3: Add the layernorm kernel

**Files:**
- Create: `python/kernels/layernorm.py`
- Modify: `python/kernels/__init__.py`

- [ ] **Step 5: Write minimal implementation**

Add a `16x16` row-wise layernorm kernel that computes:
- `mean = sum(x) * full((16, 1), 1.0 / 16.0)`
- `centered = x - mean`
- `var = sum(centered * centered) * full((16, 1), 1.0 / 16.0)`
- `normalized = centered * rsqrt(var)`

- [ ] **Step 6: Run targeted tests to verify it passes**

Run: `pytest python/tests/test_python_dsl.py -k 'layernorm' -q`
Expected: PASS

## Chunk 3: Full Verification

### Task 4: Run regression checks

**Files:**
- Modify: `pyproject.toml` only if package exports need updating

- [ ] **Step 7: Run DSL regression suite**

Run: `pytest python/tests/test_python_dsl.py -q`
Expected: PASS

- [ ] **Step 8: Run wheel regression suite**

Run: `pytest python/tests/test_wheel_build.py -q`
Expected: PASS
