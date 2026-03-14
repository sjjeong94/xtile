# Nova Matmul Design

**Goal:** Extend the `nova` dialect and `xt-to-nova` conversions with `nova.matmul`, covering `xt.matmul`.

## Scope

- Add `nova.matmul(lhs, rhs, scale, bias) -> result`
- Keep `scale` and `bias` as explicit `f32` SSA operands
- In `xt-to-nova`, materialize:
  - `%scale = arith.constant 1.0 : f32`
  - `%bias = arith.constant 0.0 : f32`
- Extend both C++ `xt-to-nova` and Python `xt.to_nova(...)`

## Conversion Rules

- `xt.matmul(lhs, rhs)` -> `nova.matmul(lhs, rhs, %scale, %bias)`
- Preserve operand order and result type
- Emit one `arith.constant` for `1.0 : f32` and one for `0.0 : f32` adjacent to the rewrite site
- Other ops stay unchanged

## Testing

- Extend lit coverage with one `xt.matmul` case that checks for `arith.constant 1.0`, `arith.constant 0.0`, and `nova.matmul`
- Extend Python DSL coverage so `xt.to_nova(xt.convert(matmul_kernel))` emits `nova.matmul`
- Extend CLI `--xt-to-nova` expectation to include `nova.matmul`
