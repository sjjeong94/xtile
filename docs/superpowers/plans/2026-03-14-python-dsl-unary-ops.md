# Python DSL Unary Ops Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all existing xt unary elementwise ops to the Python DSL surface so they can be converted via `xt.convert(...)`.

**Architecture:** Extend the Python DSL stub surface and re-exports for the unary ops already present in the xt dialect, then widen the AST parser's allowed unary op set so the existing IR builder continues to emit `xt.<op>` unchanged. Verify the feature with a focused Python DSL regression test that converts one kernel per unary op and checks the emitted MLIR.

**Tech Stack:** Python, pytest, MLIR Python bindings

---

## Chunk 1: Python DSL Unary Surface

### Task 1: Add a regression test covering all unary ops

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("op_name", [...])
def test_convert_supported_unary_kernels(op_name: str):
    ...
    assert f"xt.{op_name}" in dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py::test_convert_supported_unary_kernels -v`
Expected: FAIL because the new unary stubs and parser allowlist do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add unary stub functions and exports in `python/xtile/dsl.py` and `python/xtile/__init__.py`, then expand `_UNARY_OPS` in `python/xtile/ast_parser.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_python_dsl.py::test_convert_supported_unary_kernels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_python_dsl.py python/xtile/dsl.py python/xtile/__init__.py python/xtile/ast_parser.py docs/superpowers/plans/2026-03-14-python-dsl-unary-ops.md
git commit -m "feat: add unary ops to python dsl"
```
