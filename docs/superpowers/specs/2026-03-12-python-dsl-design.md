# xTile Python DSL Design

## Goal

Add a Python DSL that converts annotated Python kernel functions into real MLIR Python binding objects representing xt dialect programs.

## Requirements

- `xt.convert(fn)` returns an `mlir.ir.Module`
- function argument memref types come from Python annotations
- tile tensor shapes come from DSL call arguments such as `shape=(...)`
- implementation should start from the documented `add_kernel` example and remain extensible to the other examples in `xtile.md`

## Architecture

The Python DSL is split into three layers.

1. Frontend DSL API
   - `@xt.kernel`
   - `xt.memref(...)` annotation helper
   - `xt.bid(dim)`
   - `xt.load(...)`, `xt.store(...)`
   - explicit op calls such as `xt.exp`, `xt.matmul`, `xt.reduce_sum`, `xt.reshape`, `xt.transpose`
2. AST-to-typed graph lowering
   - inspect Python source with `inspect.getsource`
   - parse with `ast.parse`
   - support a restricted kernel subset and convert it into typed nodes
   - maintain symbol table entries for block ids, integer constants, memrefs, and tile values
3. Typed graph-to-MLIR lowering
   - create `mlir.ir.Context`, `Location`, and `Module`
   - build `func.func`, `arith.constant`, `xt.get_tile_block_id`, and other `xt.*` ops
   - emit operations with generic `Operation.create(...)` so Python ODS bindings are not required

## Supported Kernel Subset

Initial support targets the documented Python example and the nearby xtile examples.

- top-level functions decorated with `@xt.kernel`
- assignments to a single name
- integer constants
- tuple literals for indices, shapes, and array attrs
- `xt.bid(dim)`
- `xt.load(memref, index=(...), shape=(...), shared=...)`
- `xt.store(memref, index=(...), tile=...)`
- Python binary operators `+`, `-`, `*`
- unary and explicit xt op calls

Unsupported syntax fails conversion with a descriptive error.

- control flow
- arbitrary Python calls
- attribute mutation
- comprehensions

## Type System

Memref types are declared in annotations.

```python
def add_kernel(
    a: xt.memref("?xf32"),
    b: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    ...
```

The annotation parser records rank, dimensions, dynamic markers, and element type. These become MLIR function argument types.

Tile values use inferred tensor types.

- `xt.load(..., shape=(16,))` on `memref<?xf32>` produces `tensor<16xf32>`
- binary elementwise ops require compatible element types and use a tensor shape rule layer that can later grow to support broadcast
- op-specific rules compute output tensor types for reduce, reshape, transpose, matmul, and convolution-style ops

## MLIR Emission Strategy

The lowering stage constructs a real `mlir.ir.Module`. It inserts:

- `func.func` with memref-typed arguments
- one `xt.get_tile_block_id`
- `arith.constant` values for referenced integer constants
- generic xt dialect ops created with explicit result types and attributes
- `func.return`

`xt.dump(module)` simply stringifies the MLIR module.

## Error Handling

Conversion errors are explicit and early.

- invalid or missing memref annotations
- unsupported AST nodes
- missing `shape=` on ops that require it
- rank or element-type mismatches
- malformed attrs such as non-integer pad/stride/dilation

## Testing Strategy

Use TDD and grow from the smallest example.

1. add a failing test for the documented `add_kernel` example
2. implement the minimum path to pass it
3. extend tests for unary ops, shared loads, reduce, reshape, and transpose
4. add negative tests for unsupported syntax and type mismatches

## File Layout

- `python/xtile/__init__.py`: public API
- `python/xtile/types.py`: memref and tensor metadata parsing
- `python/xtile/dsl.py`: decorator and annotation helpers
- `python/xtile/ast_parser.py`: AST-to-typed graph conversion
- `python/xtile/ir_builder.py`: MLIR binding emission
- `python/xtile/errors.py`: conversion errors
- `python/tests/test_python_dsl.py`: behavior tests
