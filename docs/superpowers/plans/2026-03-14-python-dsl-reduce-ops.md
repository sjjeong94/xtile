# Python DSL Reduce Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `xt.sum(...)` and `xt.max(...)` to the Python DSL and lower them to `xt.reduce_sum` and `xt.reduce_max`.

**Architecture:** Keep the Python surface aligned with the user's requested names while preserving the existing MLIR dialect names through a small alias map in the AST parser. Reuse the existing unary-op IR path by mapping `sum -> reduce_sum` and `max -> reduce_max`, then verify both emitted op names and reduced tensor shapes in focused Python DSL tests.

**Tech Stack:** Python, pytest, MLIR Python bindings

---

## Chunk 1: Python DSL Reduce Aliases

### Task 1: Add a regression test covering `xt.sum` and `xt.max`

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
def test_convert_reduce_kernels():
    ...
    assert "xt.reduce_sum" in dumped
    assert "xt.reduce_max" in dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py::test_convert_supported_reduce_kernels -v`
Expected: FAIL because `xt.sum` and `xt.max` are not exported or parsed yet.

- [ ] **Step 3: Write minimal implementation**

Add `sum`/`max` stubs and exports in `python/xtile/dsl.py` and `python/xtile/__init__.py`, then map those names to `reduce_sum`/`reduce_max` in `python/xtile/ast_parser.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_python_dsl.py::test_convert_supported_reduce_kernels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_python_dsl.py python/xtile/dsl.py python/xtile/__init__.py python/xtile/ast_parser.py docs/superpowers/plans/2026-03-14-python-dsl-reduce-ops.md
git commit -m "feat: add reduce ops to python dsl"
```
