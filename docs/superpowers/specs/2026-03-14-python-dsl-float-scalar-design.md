# Python DSL Float Scalar Design

**Goal:** Allow kernels such as `layernorm_kernel` to use Python float scalar bindings like `inv_cols = 0.0625` and still convert into valid xt MLIR.

## Scope

- Extend the Python AST parser to accept float scalar assignments
- Allow binary ops between a tile tensor and a scalar float binding
- Materialize scalar floats as dense tensor constants matching the counterpart tensor

## Behavior

- Support:
  - `inv_cols = 0.0625`
  - `mean = row_sum * inv_cols`
  - `var = var_sum * inv_cols`
- Keep the emitted op surface in xt:
  - scalar float becomes `arith.constant dense<...> : tensor<...>`
  - multiplication remains `xt.mul`

## Architecture

- Add a scalar-float value kind to the parser environment
- When parsing a binary op:
  - if both sides are tiles, keep current behavior
  - if one side is a scalar float and the other is a tile, synthesize a `FullOp`
    with the tile's shape and element type, then emit the existing binary op
- Reuse the existing `FullOp` lowering path in the IR builder

## Testing

- `layernorm_kernel` must convert successfully again
- Printed MLIR must include:
  - `func.func @layernorm_kernel`
  - `xt.reduce_sum`
  - `xt.mul`
  - `arith.constant`
  - `0.062500` encoded in a dense tensor constant
