# xTile Python DSL Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an extensible Python DSL that converts annotated xt kernels into `mlir.ir.Module` objects.

**Architecture:** Parse a restricted Python AST into a typed DSL graph, then lower that graph into generic MLIR Python binding operations for `func`, `arith`, and `xt`. Start with the documented add example and extend the registry and type rules so more xtile examples can be supported incrementally.

**Tech Stack:** Python, MLIR Python bindings, pytest

---

## Chunk 1: Test Harness And First Red Test

### Task 1: Add the first failing conversion test

**Files:**
- Create: `python/tests/test_python_dsl.py`
- Create: `python/tests/conftest.py`

- [ ] **Step 1: Write the failing test**
Add a test for the documented `add_kernel` DSL example. Assert that `xt.convert(add_kernel)` returns a module whose printed MLIR contains `func.func @add_kernel`, `xt.get_tile_block_id`, two `xt.load` ops, one `xt.add`, and one `xt.store`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python:/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core python3 -m pytest python/tests/test_python_dsl.py -q`
Expected: FAIL because the `xtile` package does not exist yet.

## Chunk 2: Minimal DSL Surface

### Task 2: Add annotations and entry points

**Files:**
- Create: `python/xtile/__init__.py`
- Create: `python/xtile/dsl.py`
- Create: `python/xtile/types.py`
- Create: `python/xtile/errors.py`

- [ ] **Step 1: Write minimal implementation**
Implement `@kernel`, `memref(...)`, `convert(...)`, and `dump(...)`. Add memref annotation parsing for shapes such as `?xf32` and `2048x16xf32`.

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=python:/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core python3 -m pytest python/tests/test_python_dsl.py -q`
Expected: FAIL later in conversion because AST parsing and IR emission are still missing.

## Chunk 3: AST Parser

### Task 3: Parse the add kernel subset

**Files:**
- Create: `python/xtile/ast_parser.py`

- [ ] **Step 1: Write minimal implementation**
Support function args, local integer constants, `xt.bid(dim)`, `xt.load`, binary `+`, and `xt.store`. Build a typed graph with memref symbols, block id references, integer constants, and tile nodes.

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=python:/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core python3 -m pytest python/tests/test_python_dsl.py -q`
Expected: FAIL in MLIR emission because the graph is not lowered yet.

## Chunk 4: MLIR Lowering

### Task 4: Lower the typed graph into an MLIR module

**Files:**
- Create: `python/xtile/ir_builder.py`

- [ ] **Step 1: Write minimal implementation**
Create an `mlir.ir.Module`, emit `func.func`, `xt.get_tile_block_id`, integer constants, `xt.load`, `xt.add`, `xt.store`, and `func.return` using generic operations.

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=python:/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core python3 -m pytest python/tests/test_python_dsl.py -q`
Expected: PASS for the add example.

## Chunk 5: Extensibility Coverage

### Task 5: Add a small supported-op registry and extra tests

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Modify: `python/xtile/ast_parser.py`
- Modify: `python/xtile/ir_builder.py`

- [ ] **Step 1: Write the failing tests**
Add coverage for one unary op and one shape-transform op, plus an error test for unsupported syntax.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=python:/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core python3 -m pytest python/tests/test_python_dsl.py -q`
Expected: FAIL because those operations are not yet modeled.

- [ ] **Step 3: Write minimal implementation**
Add an op registry for unary and explicit xt ops, implement reshape and transpose result-type handling, and raise descriptive errors for unsupported AST.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=python:/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core python3 -m pytest python/tests/test_python_dsl.py -q`
Expected: PASS
