# Matmul Kernel Conversion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `xt.convert(...)` to convert the `matmul_kernel` defined in `dsl_example.py`.

**Architecture:** Extend the Python DSL parser to recognize the Python `@` operator as a dedicated matmul node with explicit shape validation, then lower that node to the existing `xt.matmul` dialect op in the MLIR builder. Keep the change narrow by reusing the current load/store flow and verifying behavior with a focused regression test that imports the documented example kernel.

**Tech Stack:** Python, pytest, MLIR Python bindings, xt dialect

---

## Chunk 1: Python DSL Matmul Support

### Task 1: Regression test for documented matmul example

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

```python
def test_convert_matmul_kernel():
    module = xt.convert(matmul_kernel)
    dumped = xt.dump(module)

    assert "func.func @matmul_kernel" in dumped
    assert dumped.count("xt.load") == 2
    assert "xt.matmul" in dumped
    assert "xt.store" in dumped
    assert "tensor<64x64xf32>" in dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py::test_convert_matmul_kernel -v`
Expected: FAIL because the parser rejects the `@` operator.

- [ ] **Step 3: Write minimal implementation**

Add a dedicated `MatmulOp` AST node, parse `ast.MatMult` in `python/xtile/ast_parser.py`, validate `(M, K) @ (K, N) -> (M, N)`, and emit `xt.matmul` from `python/xtile/ir_builder.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_python_dsl.py::test_convert_matmul_kernel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_python_dsl.py python/xtile/ast_parser.py python/xtile/ir_builder.py dsl_example.py docs/superpowers/plans/2026-03-14-matmul-kernel-conversion.md
git commit -m "feat: support matmul conversion in python dsl"
```

### Task 2: Keep the example entrypoint aligned with the new support

**Files:**
- Modify: `dsl_example.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Update the example entrypoint**

```python
def main() -> None:
    module = xt.convert(matmul_kernel)
    print(xt.dump(module))
```

- [ ] **Step 2: Run focused verification**

Run: `pytest python/tests/test_python_dsl.py -v`
Expected: PASS
