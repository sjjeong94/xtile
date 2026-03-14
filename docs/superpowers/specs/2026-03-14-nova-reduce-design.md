# Nova Reduce Design

**Goal:** Extend the `nova` dialect and `xt-to-nova` conversions with `nova.reduce`, covering `xt.reduce_sum` and `xt.reduce_max`.

## Scope

- Add `nova.reduce(input) -> result`
- Add one attribute:
  - `mode = 0` for sum
  - `mode = 1` for max
- Keep `mode` typed as `i32` for consistency with the existing nova ops
- Extend both C++ `xt-to-nova` and Python `xt.to_nova(...)`

## Conversion Rules

- `xt.reduce_sum` -> `nova.reduce(mode = 0 : i32)`
- `xt.reduce_max` -> `nova.reduce(mode = 1 : i32)`
- Preserve operand order and result type
- Other ops stay unchanged

## Testing

- Extend lit coverage with `xt.reduce_sum` and `xt.reduce_max`
- Extend Python DSL coverage so `xt.to_nova(xt.convert(softmax_kernel))` emits:
  - one `nova.reduce` with `mode = 1 : i32`
  - one `nova.reduce` with `mode = 0 : i32`
- Extend CLI `--xt-to-nova` expectation to include `nova.reduce`
