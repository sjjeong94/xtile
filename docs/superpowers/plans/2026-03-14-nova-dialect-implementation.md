# Nova Dialect Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `nova` dialect, implement an `xt-to-nova` conversion pass for binary tile ops, and expose the conversion through the Python package.

**Architecture:** Define `nova` with two generic binary tensor ops and rewrite `xt.add`/`xt.mul`/`xt.sub` into `nova.elementwise` or `nova.broadcast` based on operand/result shapes. Keep frontend conversion in `xt`, then expose backend conversion through a dedicated C++ pass and a small Python helper that runs that pass.

**Tech Stack:** MLIR ODS/TableGen, C++ dialect/pass registration, MLIR rewrite patterns, MLIR Python bindings, `pytest`, `lit`

---

## Chunk 1: Tests First

### Task 1: Add lit coverage for xt-to-nova rewriting

**Files:**
- Modify: `test/CMakeLists.txt`
- Create: `test/xt/to_nova.mlir`

- [ ] **Step 1: Write the failing test**

```mlir
// RUN: xt-opt --xt-to-nova %s | FileCheck %s

module {
  func.func @broadcast_sub(%a: tensor<16x16xf32>, %b: tensor<16x1xf32>) -> tensor<16x16xf32> {
    %0 = "xt.sub"(%a, %b) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @elementwise_mul(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "xt.mul"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}

// CHECK-LABEL: func.func @broadcast_sub
// CHECK: "nova.broadcast"(%arg0, %arg1) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 3 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}>
// CHECK-LABEL: func.func @elementwise_mul
// CHECK: "nova.elementwise"(%arg0, %arg1) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}>
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target check-xt`
Expected: FAIL because `--xt-to-nova` and/or the `nova` dialect are not defined.

- [ ] **Step 3: Write minimal implementation**

Add the test file and include it in the lit suite if needed by the existing layout.

- [ ] **Step 4: Run test to verify it still fails for the right reason**

Run: `cmake --build build --target check-xt`
Expected: FAIL because the code path is still unimplemented, not because the test file is malformed.

### Task 2: Add Python regression tests for explicit nova conversion

**Files:**
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_convert_then_to_nova_broadcast_binary_kernel():
    module = xt.convert(broadcast_sub_kernel)
    xt.to_nova(module)
    dumped = xt.dump(module)

    assert "nova.broadcast" in dumped
    assert "mode = 3 : i32" in dumped
    assert "xt.sub" not in dumped


@xt.kernel
def elementwise_mul_kernel(
    a: xt.memref("?xf32"),
    b: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    lhs = xt.load(a, index=(block_id,), shape=(16, 16))
    rhs = xt.load(b, index=(block_id,), shape=(16, 16))
    prod = lhs * rhs
    xt.store(result, index=(block_id,), tile=prod)


def test_convert_then_to_nova_elementwise_kernel():
    module = xt.convert(elementwise_mul_kernel)
    xt.to_nova(module)
    dumped = xt.dump(module)

    assert "nova.elementwise" in dumped
    assert "mode = 2 : i32" in dumped
    assert "xt.mul" not in dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py -k nova -v`
Expected: FAIL because `xt.to_nova` does not exist yet.

- [ ] **Step 3: Keep existing xt conversion behavior covered**

Ensure the pre-existing tests still assert `xt.sub` / `xt.mul` directly from `xt.convert(...)`, so frontend behavior stays unchanged.

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest python/tests/test_python_dsl.py -k nova -v`
Expected: FAIL only on missing nova conversion support.

## Chunk 2: C++ Nova Dialect And Pass

### Task 3: Define and register the nova dialect

**Files:**
- Create: `include/nova/NovaDialect.td`
- Create: `include/nova/NovaDialect.h`
- Create: `include/nova/NovaOps.td`
- Create: `include/nova/CMakeLists.txt`
- Create: `lib/nova/NovaDialect.cpp`
- Create: `lib/nova/CMakeLists.txt`
- Modify: `include/CMakeLists.txt`
- Modify: `lib/CMakeLists.txt`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Implement ODS declarations**
- [ ] **Step 2: Generate/register the dialect and ops**
- [ ] **Step 3: Build to catch TableGen or registration issues**

Run: `cmake --build build --target xt-opt`
Expected: PASS

### Task 4: Add the xt-to-nova pass

**Files:**
- Modify: `include/xt/XTPasses.td`
- Modify: `include/xt/XTPasses.h`
- Create: `lib/xt/XTToNova.cpp`
- Modify: `lib/xt/CMakeLists.txt`
- Modify: `xt-opt/xt-opt.cpp`

- [ ] **Step 1: Implement rewrite patterns for `xt.add`, `xt.mul`, `xt.sub`**
- [ ] **Step 2: Select `nova.elementwise` vs `nova.broadcast` from ranked tensor shapes**
- [ ] **Step 3: Populate default attrs and replace matched ops**
- [ ] **Step 4: Register the pass and dialect dependencies**

- [ ] **Step 5: Run the lit suite**

Run: `cmake --build build --target check-xt`
Expected: PASS for the new `to_nova` coverage.

## Chunk 3: Python Integration

### Task 5: Expose `xtile.to_nova(...)`

**Files:**
- Modify: `python/xtile/__init__.py`

- [ ] **Step 1: Write minimal helper**

```python
from mlir import passmanager


def to_nova(module: ir.Module) -> ir.Module:
    with module.context:
        pm = passmanager.PassManager.parse("builtin.module(func.func(xt-to-nova))")
        pm.run(module.operation)
    return module
```

- [ ] **Step 2: Export it from the package**
- [ ] **Step 3: Run targeted Python tests**

Run: `pytest python/tests/test_python_dsl.py -k nova -v`
Expected: PASS

## Chunk 4: Full Verification

### Task 6: Run focused regression coverage

**Files:**
- Modify: none unless regressions are found

- [ ] **Step 1: Run Python DSL regression tests**

Run: `pytest python/tests/test_python_dsl.py -v`
Expected: PASS

- [ ] **Step 2: Run MLIR lit tests**

Run: `cmake --build build --target check-xt`
Expected: PASS

- [ ] **Step 3: Inspect resulting IR manually if needed**

Run: `xt-opt --xt-to-nova test/xt/to_nova.mlir`
Expected: only targeted binary ops are rewritten to `nova.*`.
