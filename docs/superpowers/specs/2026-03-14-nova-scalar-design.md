# Nova Scalar Design

## Goal

Add a `nova.scalar` operation for binary tensor ops whose right-hand side is a compile-time `f32` constant, and teach both the C++ `xt-to-nova` pass and the Python `xt.to_nova()` rewrite to emit it.

## Scope

This change is intentionally narrow:

- `nova.scalar` is a new Nova dialect op.
- `mode` values match `nova.elementwise` exactly: `add=1`, `mul=2`, `sub=3`.
- The op is only produced when the right-hand side is a compile-time constant.
- Non-constant scalar-like tensors remain unchanged and continue to stay as `xt.*`.

## Operation Shape

The new op has one tensor operand and one tensor result:

```mlir
%1 = nova.scalar(%0) {mode = 1 : i32, rhs = 0.125 : f32} : tensor<64x32xf32>
```

Semantics:

- `input` is the tensor being transformed.
- `rhs` is the scalar compile-time constant applied elementwise.
- `mode` chooses add/mul/sub using the same encoding as existing Nova binary ops.
- Result type matches the input tensor type.

## Lowering Rules

`xt-to-nova` rewrites `xt.add`, `xt.mul`, and `xt.sub` to `nova.scalar` only when all of the following are true:

1. The right-hand side operand is defined by `arith.constant`.
2. The constant value can be read as an `f32` scalar or splat tensor element.
3. The left-hand side and result are ranked tensor types.

When those conditions are not met, existing behavior remains:

- normal tensor/tensor ops still lower to `nova.elementwise` or `nova.broadcast`
- scalar-like but non-constant RHS stays as `xt.*`

## Implementation Notes

- Extend `include/nova/NovaOps.td` with a dedicated unary-style op instead of overloading `nova.elementwise`.
- Keep the C++ and Python conversion logic aligned so `xt-opt --xt-to-nova` and `xt.to_nova()` produce the same form.
- Prefer minimal helpers for extracting `f32` constants from `arith.constant`.

## Testing

Add regression tests for:

- MLIR pass lowering to `nova.scalar`
- Existing scalar-like non-constant case remaining as `xt.mul`
- Python `xt.to_nova()` producing `nova.scalar`
- Existing elementwise/broadcast lowering continuing to pass
