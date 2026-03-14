# Nova Optimize Design

## Goal

Add a `--nova-optimize` pass that performs simple Nova IR peephole rewrites, starting with folding `nova.scalar` into `nova.broadcast` and `nova.elementwise`.

## Scope

The first optimization is intentionally narrow:

- Target ops: `nova.broadcast`, `nova.elementwise`
- Fold source: a `nova.scalar` defining either `lhs` or `rhs`
- Supported scalar modes:
  - `mode = 1` (`add`): keep scale unchanged, add `scalar_rhs * current_scale` into bias
  - `mode = 2` (`mul`): multiply current scale by `scalar_rhs`, keep bias unchanged
- Unsupported scalar modes are left unchanged for now

## Rewrite Rule

Given:

```mlir
%s = "nova.scalar"(%x) <{mode = ..., rhs = cst : f32}> : (tensor<...>) -> tensor<...>
%y = "nova.broadcast"(%a, %s) <{... rhs_s = s, rhs_b = b ...}> : ...
```

The pass rewrites `%y` to use `%x` directly and updates the affected side's attributes:

- For add mode:
  - new scale = old scale
  - new bias = old bias + cst * old scale
- For mul mode:
  - new scale = old scale * cst
  - new bias = old bias

The same logic applies symmetrically to the `lhs` side.

If both operands are defined by `nova.scalar`, the pass folds both in one rewrite.

## Pass Structure

- Introduce Nova pass declarations in `include/nova/NovaPasses.td` and `include/nova/NovaPasses.h`
- Implement the pass in `lib/nova/NovaOptimize.cpp`
- Register Nova passes in `xt-opt` alongside existing XT passes

## Testing

Add lit coverage for:

- folding a scalar on `rhs` into `nova.broadcast`
- folding a scalar on `lhs` into `nova.elementwise`
- folding both sides in one op
- leaving `mode = 3` scalar inputs unchanged
